"""Where the G5 training step's memory peak falls, by phase, at one sequence length.

Asked 2026-09-25 before trying chunked cross-entropy (David approved the idea): chunking
the loss can only lower the step's peak if the peak falls while the full logits exist.

Config: Route 1 + lean scan (installed here) + whatever wraps this script (attn_bf16.py
for the configuration that passed G2 and G5); placement A, PEFT, optimizer and data as
g5_train_step.py (records shuffled with random.Random(0), rendered with Lightning's
template and the lab's TOOLS, packed into L-token sequences, labels = inputs).

For each loss mode, "hf" (the stock forward) then "chunked" (chunked_ce.py), in one load:
one warm-up step on sequence 0, then one recorded step on sequence 1, with
  - torch peak allocated per phase (the peak counter is reset at each boundary):
    body_forward (embedding to the final norm), loss_forward (lm_head, fp32 upcast,
    cross-entropy), loss_backward (until the final norm's output has its gradient),
    body_backward (recompute + backward of the 52 blocks), optimizer;
  - a CUDA memory history of that step: when the peak falls and which allocations made
    during the step are live then, grouped by the innermost probe/transformers/peft/bnb
    frame.
Before the chunked steps, an exactness check of the loss head alone on real final hidden
states (no grad through the body): stock lm_head + ForCausalLMLoss against chunked_loss.
Lines CHOSEN before the first run: |loss difference| <= 1e-4 nats; gradient w.r.t. the
hidden states max |difference| <= 1e-2 x max |stock gradient| and cosine >= 0.9999.

Run (laptop, max-power), e.g. through the attention-bf16 wrapper:
  ... --entrypoint python3 gpu-lab:training /probes/attn_bf16.py g5_mem_phases.py LABEL [2048] [256]
Mounts as g5_train_step.py (probes, results at /out, ~/gpu-lab/training at /gpulab/training).
"""
import gc
import json
import os
import random
import sys
import time

import bitsandbytes as bnb
import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import chunked_ce  # noqa: E402
import lean_scan  # noqa: E402
import route1  # noqa: E402
from route1 import REPO, REV, gib  # noqa: E402

sys.path.insert(0, "/gpulab/training")
from tools import TOOLS  # noqa: E402

label = sys.argv[1]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 256
DATASET = "/gpulab/training/research_dataset_v3.json"
TARGETS = r".*\.mixer\.(q_proj|k_proj|v_proj|o_proj|in_proj)$|.*\.mixer\.shared_experts\.(up_proj|down_proj)$"
SITE_KEYS = ("/probes/", "transformers/", "peft/", "bitsandbytes/")

lean_scan.install()
sampler = route1.Sampler(interval=0.1)
sampler.start()
torch.cuda.init()
t_start = time.time()
res = {"label": label, "seq": L, "chunk": CHUNK, "model": REPO, "revision": REV,
       "platform_profile": os.environ.get("PLATFORM_PROFILE", "not passed"),
       "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
       "transformers": transformers.__version__, "bitsandbytes": bnb.__version__, "peft": peft.__version__}


def dump():
    res["time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    res["wall_s"] = round(time.time() - t_start, 1)
    res["phase_peaks"] = sampler.report()
    res["abort"] = sampler.abort
    res["lean_scan"] = lean_scan.report()
    res["chunked_ce"] = chunked_ce.report()
    with open(f"/out/{label}.json", "w") as f:
        json.dump(res, f, indent=1)


def free():
    gc.collect()
    torch.cuda.empty_cache()


# ---- data (g5_train_step.py's rendering and packing, same order)
tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
records = json.load(open(DATASET))
random.Random(0).shuffle(records)


def lightning_messages(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                c = json.loads(json.dumps(c))
                fn = c.get("function", c)
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"])
                calls.append(c)
            m["tool_calls"] = calls
        out.append(m)
    return out


stream = []
for r in records:
    try:
        text = tok.apply_chat_template(lightning_messages(r["messages"]), tools=TOOLS, tokenize=False)
    except Exception:
        continue
    stream.extend(tok(text, add_special_tokens=False)["input_ids"])
seqs = [stream[k * L:(k + 1) * L] for k in range(2)]

# ---- model (as g5_train_step.py)
sampler.phase = "load"
model, info, load_s = route1.load()
res["load"] = {"seconds": round(load_s, 1), "hook": route1.hook_summary(), "missing_keys": len(info["missing_keys"])}
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                                         task_type="CAUSAL_LM", target_modules=TARGETS))
trainable = [p for p in model.parameters() if p.requires_grad]
opt = bnb.optim.PagedAdamW8bit(trainable, lr=1e-4, weight_decay=0.0)
model.train()
base = model.base_model.model  # NemotronHForCausalLM
res["forward_is_instance_attr"] = "forward" in vars(base)
free()
res["idle_after_peft_torch_allocated_GiB"] = gib(torch.cuda.memory_allocated())
dump()

# ---- phase marks
cur = {"state": None, "marks": []}


def mark(name):
    cur["marks"].append({"phase": name, "peak_GiB": gib(torch.cuda.max_memory_allocated()),
                         "allocated_at_end_GiB": gib(torch.cuda.memory_allocated())})
    torch.cuda.reset_peak_memory_stats()


def lm_head_pre(mod, args):
    if cur["state"] == "body_forward":
        mark("body_forward")
        cur["state"] = "loss_forward"


def norm_out(mod, args, out):
    if cur["state"] == "body_forward":
        out.register_hook(lambda g: (mark("loss_backward"), cur.__setitem__("state", "body_backward"))[1])


base.lm_head.register_forward_pre_hook(lm_head_pre)
base.model.norm_f.register_forward_hook(norm_out)


def step(ids, instrument):
    x = torch.tensor([ids], device=0)
    if instrument:
        cur["marks"] = []
        torch.cuda.reset_peak_memory_stats()
        cur["state"] = "body_forward"
    out = model(input_ids=x, labels=x, use_cache=False)
    loss = out.loss
    if instrument:
        mark("loss_forward")
        cur["state"] = "loss_backward"
    loss.backward()
    if instrument:
        mark("body_backward")
        cur["state"] = "optimizer"
    opt.step()
    opt.zero_grad(set_to_none=True)
    if instrument:
        mark("optimizer")
        cur["state"] = None
    return loss.item(), out.logits is None


def site(frames):
    rel = [f for f in frames if any(k in f["filename"] for k in SITE_KEYS)]
    if not rel:
        rel = frames[:1]
    return " < ".join(f"{os.path.basename(f['filename'])}:{f['line']} {f['name']}" for f in rel[:2])


def analyze(snap, base_alloc):
    tr = snap["device_traces"][0]
    size_of, level, best, best_i = {}, 0, 0, -1
    for i, e in enumerate(tr):
        if e["action"] == "alloc":
            size_of[e["addr"]] = e["size"]
            level += e["size"]
        elif e["action"] == "free_completed" and e["addr"] in size_of:
            level -= size_of.pop(e["addr"])
        if level > best:
            best, best_i = level, i
    live = {}
    for e in tr[:best_i + 1]:
        if e["action"] == "alloc":
            live[e["addr"]] = e
        elif e["action"] == "free_completed":
            live.pop(e["addr"], None)
    groups = {}
    for e in live.values():
        g = groups.setdefault(site(e.get("frames", [])), {"n": 0, "bytes": 0})
        g["n"] += 1
        g["bytes"] += e["size"]
    top = sorted(groups.items(), key=lambda kv: -kv[1]["bytes"])[:15]
    peak_frames = tr[best_i].get("frames", []) if best_i >= 0 else []
    return {"trace_events": len(tr), "peak_event_index": best_i,
            "step_allocations_live_at_peak_GiB": gib(best),
            "allocated_before_step_GiB": gib(base_alloc),
            "implied_peak_GiB": gib(base_alloc + best),
            "peak_event_site": site(peak_frames),
            "live_at_peak_by_site": [{"site": s, "n": g["n"], "GiB": gib(g["bytes"])} for s, g in top]}


def run_mode(mode):
    out = {}
    res["modes"][mode] = out
    free()
    sampler.phase = f"{mode}_warmup"
    loss, no_logits = step(seqs[0], False)
    out["warmup"] = {"loss": round(loss, 4), "logits_none": no_logits}
    dump()
    if sampler.abort:
        return
    free()
    sampler.phase = f"{mode}_recorded"
    base_alloc = torch.cuda.memory_allocated()
    torch.cuda.memory._record_memory_history(enabled="all", context="alloc", stacks="python")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        loss, no_logits = step(seqs[1], True)
        out["recorded"] = {"loss": round(loss, 4), "logits_none": no_logits,
                           "seconds_with_history": round(time.perf_counter() - t0, 2), "marks": cur["marks"],
                           "step_peak_GiB": max(m["peak_GiB"] for m in cur["marks"])}
    except torch.OutOfMemoryError as e:
        out["recorded"] = {"oom": str(e)[:600], "marks": cur["marks"]}
        opt.zero_grad(set_to_none=True)
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    out["snapshot"] = analyze(snap, base_alloc)
    del snap
    dump()


# ---- run
res["modes"] = {}
run_mode("hf")
if not sampler.abort:
    chunked_ce.install(CHUNK)
    free()
    sampler.phase = "exactness"
    ids = torch.tensor([seqs[1]], device=0)
    with torch.no_grad():
        hid = base.model(input_ids=ids, use_cache=False)[0]
    h1 = hid.detach().requires_grad_(True)
    logits = base.lm_head(h1).float()
    loss_hf = base.loss_function(logits, ids, base.vocab_size)
    loss_hf.backward()
    del logits
    h2 = hid.detach().requires_grad_(True)
    loss_ch = chunked_ce.chunked_loss(base.lm_head, h2, ids, CHUNK)
    loss_ch.backward()
    g1, g2 = h1.grad.float(), h2.grad.float()
    diff = (g1 - g2).abs().max().item()
    gmax = g1.abs().max().item()
    cos = torch.nn.functional.cosine_similarity(g1.flatten(), g2.flatten(), dim=0).item()
    ex = {"loss_hf": loss_hf.item(), "loss_chunked": loss_ch.item(),
          "loss_abs_diff": abs(loss_hf.item() - loss_ch.item()),
          "grad_max_abs_diff": diff, "grad_max_abs_ref": gmax, "grad_rel": diff / gmax, "grad_cosine": cos,
          "grad_bitwise_equal": bool(torch.equal(h1.grad, h2.grad))}
    ex["pass"] = ex["loss_abs_diff"] <= 1e-4 and ex["grad_rel"] <= 1e-2 and cos >= 0.9999
    res["exactness"] = ex
    del hid, h1, h2, g1, g2, loss_hf, loss_ch
    dump()
    run_mode("chunked")

sampler.stop = True
dump()
print(json.dumps({"exactness": res.get("exactness"),
                  "modes": {m: {"recorded": {k: v for k, v in o.get("recorded", {}).items()},
                                "peak": o.get("snapshot", {}).get("implied_peak_GiB"),
                                "peak_site": o.get("snapshot", {}).get("peak_event_site")}
                            for m, o in res["modes"].items()}}, indent=1))

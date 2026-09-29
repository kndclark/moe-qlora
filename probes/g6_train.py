"""G6 (plan.md): train placement A on research_dataset_v3, like for like with the Qwen3-8B v3 adapter.

Kept from gpu-lab training/qlora.py, which trained v3 (adapters/qwen3-8b-research-v3/
train-report.json: 950 records, 2 epochs, 238 steps, lr 1e-4):
  - the data: every record of research_dataset_v3.json, rendered with the model's chat template
    and the lab's TOOLS, truncated at 1024 tokens; loss only on assistant tokens, from each
    "<|im_start|>assistant\\n" to and including its <|im_end|>; records marked thinking "off" get
    the empty think block before every tool call, masked as prompt (qlora.py:149-190). Lightning's
    template already writes "<think></think>" before every assistant turn, and vLLM's thinking-off
    prompt ends with exactly that (enable_thinking=False generation prompt, checked 2026-09-27),
    so for "off" records that block is masked; qlora.py's Qwen-string insertion finds nothing
    to insert here. Default records train it, as qlora.py trained Qwen's own empty block;
  - max length 1024, qlora.py's default. Both templates cut long records: 439 of 950 here,
    375 with Qwen3-8B's (if v3 ran at the default, which its report does not record);
  - LoRA r=16, alpha=32, dropout 0.05, bias none;
  - paged AdamW 8-bit, lr 1e-4, weight decay 0, cosine schedule with warmup = round(3% of
    steps), gradient-norm clip 1.0 (Trainer's default), seed 0, a checkpoint at each epoch;
  - 8 records per optimizer step, the loss normalised over every assistant token in the step,
    as transformers 5.x's Trainer does with num_items_in_batch. That makes 1 x 8 here the same
    objective as qlora.py's 2 x 4, not an approximation of it.
Forced by the model (plan.md G5 and "After the verdict"):
  - placement A targets, Route 1 4-bit load, attention kept bf16 (run through attn_bf16.py),
    lean scan, chunked cross-entropy: F1, the configuration that passes G5 on the laptop;
  - batch 1, the size G5 measured; no fp32 upcast of non-4-bit parameters (G5's choice;
    qlora.py upcasts by default, and whether v3 ran with --no-fp32-upcast is not recorded).
Record order: a fresh permutation per epoch from torch.Generator(seed + epoch). Trainer's
sampler draws a different order from the same data.

Thermal guard: route1.Sampler, as in G5 unless GUARD says otherwise:
  GUARD=g5  (default) abort on sw_thermal, hw_thermal or hw_power_brake, or at 87 C
  GUARD=hw  abort on hw_thermal or hw_power_brake, or at 90 C; sw_thermal (the card holding
            its own 87 C target) is counted, not fatal
The guard stops the run between optimizer steps; the adapter so far is saved as -partial.

Render: RENDER=g6 (default) is G6 as run. RENDER=think is G6r (plan.md "G6R"): default
records start each assistant turn with "<think>\n" masked, as the thinking-on prompt ends,
and train "</think>" + content; "off" records are unchanged. One sequence per record, so
history turns read "<think>\n</think>" where the server sends "<think></think>" (design A,
David 2026-09-27). parity() checks every trained turn against the server's prompt; DRY_RUN
prints it for both renders, and RENDER=think will not train if it fails.

Run (laptop):
  probes/gpurun.sh g6-train /probes/attn_bf16.py g6_train.py g6-train
  RENDER=think GUARD=hw probes/gpurun.sh g6r-train /probes/attn_bf16.py g6_train.py g6r-train
  DATASET=/out/<file>.json trains on another file (a container path; /out is results/);
  unset, it is v3, as G6 and G6r ran.
  MAX_LEN=2048 caps at the length Qwen v3 actually trained at (qlora.py --max-len 2048,
  plan.md "G6P2048"); unset, it is 1024, as G6, G6r and G6p ran.
  DRY_RUN=1 renders and masks the data, prints samples and stops before loading the model.
Output: /out/<label>-adapter/ (checkpoint-N per epoch, final adapter at the top),
/out/<label>.json (rewritten every step).
"""
import gc
import json
import math
import os
import sys
import time

import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, "/gpulab/training")
import chunked_ce  # noqa: E402
import lean_scan  # noqa: E402
import route1  # noqa: E402
from route1 import REPO, REV, gib  # noqa: E402
from tools import TOOLS  # noqa: E402
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup  # noqa: E402

label = sys.argv[1]
DRY = os.environ.get("DRY_RUN") == "1"
GUARD = os.environ.get("GUARD", "g5")
RENDER = os.environ.get("RENDER", "g6")
if RENDER not in ("g6", "think", "trace"):
    sys.exit(f"RENDER must be g6, think or trace, not {RENDER}")
DATASET = os.environ.get("DATASET") or "/gpulab/training/research_dataset_v3.json"
TARGETS = r".*\.mixer\.(q_proj|k_proj|v_proj|o_proj|in_proj)$|.*\.mixer\.shared_experts\.(up_proj|down_proj)$"
EPOCHS, LR, PER_STEP, RANK, SEED = 2, 1e-4, 8, 16, 0
MAX_LEN = int(os.environ.get("MAX_LEN") or 1024)
OUT = f"/out/{label}-adapter"

# ---- data (qlora.py's encode, with Lightning's tokenizer)
tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
records = json.load(open(DATASET))
assistant_header = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
im_end = tok.encode("<|im_end|>", add_special_tokens=False)
assert len(im_end) == 1, im_end
im_end = im_end[0]
empty_think = "<think>\n\n</think>\n\n"
empty_think_ids = tok.encode(empty_think, add_special_tokens=False)
lightning_empty_think = tok.encode("<think></think>", add_special_tokens=False)
# RENDER=think (G6r): the thinking-on generation prompt ends "<|im_start|>assistant\n<think>\n",
# so default records get that opening masked as prompt and train "</think>" + content.
# G6 trained "<think></think>" + content from "assistant\n", a point the server's prompt
# has already passed.
think_open = "<think>\n"
think_open_ids = tok.encode(think_open, add_special_tokens=False)


def lightning_messages(msgs):
    # g5_train_step.py: Lightning's template wants tool-call arguments as objects.
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


def tools_for(record):
    # G6q: a record may carry tools the eval offers beside training's two (the promql
    # set's promql tool); records without "extra_tools" render exactly as before.
    return TOOLS + record.get("extra_tools", [])


def encode(record, how="g6"):
    text = tok.apply_chat_template(lightning_messages(record["messages"]), tools=tools_for(record), tokenize=False)
    off = record.get("thinking") == "off"
    inserted = 0
    if off:
        before = text
        text = text.replace("<|im_start|>assistant\n<tool_call>",
                            "<|im_start|>assistant\n" + empty_think + "<tool_call>")
        inserted = (len(text) - len(before)) // len(empty_think)
    elif how == "think":
        # Every turn, the history ones too: one sequence per record (design A), so earlier
        # turns read "<think>\n</think>" where the server's history has "<think></think>".
        text = text.replace("<|im_start|>assistant\n<think></think>",
                            "<|im_start|>assistant\n" + think_open + "</think>")
    ids = tok(text, truncation=True, max_length=MAX_LEN, add_special_tokens=False)["input_ids"]
    full_len = len(tok(text, add_special_tokens=False)["input_ids"])
    labels = [-100] * len(ids)
    masked_think = 0
    turns = []  # (assistant header position, first trained position) per turn
    i = 0
    while i < len(ids):
        if ids[i:i + len(assistant_header)] == assistant_header:
            start = i + len(assistant_header)
            pres = ((empty_think_ids, lightning_empty_think) if off
                    else (think_open_ids,) if how == "think" else ())
            for pre in pres:
                if ids[start:start + len(pre)] == pre:
                    start += len(pre)
                    masked_think += 1
                    break
            turns.append((i, start))
            end = start
            while end < len(ids) and ids[end] != im_end:
                end += 1
            if end < len(ids):
                end += 1
            for j in range(start, end):
                labels[j] = ids[j]
            i = end
        else:
            i += 1
    return {"ids": ids, "labels": labels, "truncated": full_len > MAX_LEN, "inserted": inserted,
            "masked_think": masked_think, "thinking": record.get("thinking"), "turns": turns}


def parity(recs, encoded):
    """For every trained turn, the tokens before its trained span against the prompt the
    eval server builds for that turn: bench/research_eval.py sends back the history (visible
    text + tool call, never reasoning) and asks for add_generation_prompt with
    enable_thinking on, or off for "off" records. The boundary (header to first trained
    token) must match on every turn; history differences before it are counted.
    """
    st = {"turns": 0, "cut": 0, "boundary_ok": 0, "boundary_bad": 0, "exact": 0,
          "history_diff_turns": 0, "history_diff_tokens": 0, "bad_examples": []}
    for rec, d in zip(recs, encoded):
        msgs = lightning_messages(rec["messages"])
        ks = [j for j, m in enumerate(msgs) if m["role"] == "assistant"]
        for k, (h, start) in zip(ks, d["turns"]):
            st["turns"] += 1
            if start >= len(d["ids"]):  # truncation cut the turn before anything is trained
                st["cut"] += 1
                continue
            srv = tok(tok.apply_chat_template(msgs[:k], tools=tools_for(rec), tokenize=False, add_generation_prompt=True,
                                              enable_thinking=rec.get("thinking") != "off"),
                      add_special_tokens=False)["input_ids"]
            pre = d["ids"][:start]
            sh = max(j for j in range(len(srv)) if srv[j:j + len(assistant_header)] == assistant_header)
            if srv[sh:] == pre[h:]:
                st["boundary_ok"] += 1
            else:
                st["boundary_bad"] += 1
                if len(st["bad_examples"]) < 3:
                    st["bad_examples"].append({"record": rec.get("id", recs.index(rec)), "turn": k,
                                               "server": tok.decode(srv[sh:]), "train": tok.decode(pre[h:])})
            if srv == pre:
                st["exact"] += 1
            elif srv[:sh] != pre[:h]:
                st["history_diff_turns"] += 1
                st["history_diff_tokens"] += len(pre[:h]) - len(srv[:sh])
    st["pass"] = st["boundary_bad"] == 0 and st["boundary_ok"] > 0
    return st


def encode_trace(record, st):
    """RENDER=trace (G6t): a record with an accepted base-Lightning trace becomes one
    sequence per assistant turn. Prompt: the template's thinking-on generation prompt over
    the history exactly as the eval harness sends it (visible text + call, no reasoning),
    masked. Trained: base's own completion text for that turn, reasoning included, then
    <|im_end|>. The guard: the prompt must have the token count vLLM's /tokenize gave
    for that turn when the trace was collected; completion counts are reported."""
    tr = record["trace"]
    out = []
    for k, text in enumerate(tr["texts"]):
        if "train_turns" in tr and k not in tr["train_turns"]:  # G6u: a turn with empty reasoning
            continue
        hist = lightning_messages(tr["messages"][:1 + 2 * k])
        prompt = tok.apply_chat_template(hist, tools=tools_for(record), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(text, add_special_tokens=False)["input_ids"] + [im_end]
        st["turns"] += 1
        if len(p_ids) == tr["prompt_tokens"][k]:
            st["prompt_count_ok"] += 1
        else:
            st["prompt_count_bad"] += 1
            if len(st["bad_examples"]) < 3:
                st["bad_examples"].append({"index": tr["index"], "turn": k, "train": len(p_ids),
                                           "server": tr["prompt_tokens"][k]})
        st["completion_count_equal"] += len(c_ids) == tr["completion_tokens"][k]
        ids = p_ids + c_ids
        out.append({"ids": ids[:MAX_LEN], "labels": ([-100] * len(p_ids) + c_ids)[:MAX_LEN],
                    "truncated": len(ids) > MAX_LEN, "inserted": 0, "masked_think": 0,
                    "thinking": "trace", "turns": []})
    return out


trace_parity = {"turns": 0, "prompt_count_ok": 0, "prompt_count_bad": 0, "completion_count_equal": 0,
                "bad_examples": []}
if RENDER == "trace":
    plain = [r for r in records if not r.get("trace")]
    data = [encode(r, "think") for r in plain]
    for r in records:
        if r.get("trace"):
            data.extend(encode_trace(r, trace_parity))
    trace_parity["records_with_trace"] = len(records) - len(plain)
    trace_parity["pass"] = trace_parity["prompt_count_bad"] == 0 and trace_parity["turns"] > 0
else:
    plain = records
    data = [encode(r, RENDER) for r in records]
n_items = [sum(l != -100 for l in d["labels"]) for d in data]
lens = sorted(len(d["ids"]) for d in data)
res = {"label": label, "model": REPO, "revision": REV, "dataset": DATASET, "guard": GUARD,
       "config": {"epochs": EPOCHS, "lr": LR, "records_per_step": PER_STEP, "batch": 1,
                  "max_len": MAX_LEN, "rank": RANK, "alpha": 2 * RANK, "dropout": 0.05,
                  "seed": SEED, "targets": TARGETS, "clip": 1.0, "warmup_frac": 0.03},
       "data": {"records": len(data), "tokens": sum(lens), "assistant_tokens": sum(n_items),
                "p50": lens[len(lens) // 2], "p90": lens[int(len(lens) * 0.9)], "max": lens[-1],
                "truncated": sum(d["truncated"] for d in data),
                "no_assistant_tokens": sum(n == 0 for n in n_items),
                "off_records": sum(d["thinking"] == "off" for d in data),
                "off_records_with_insert": sum(d["thinking"] == "off" and d["inserted"] > 0 for d in data),
                "off_records_think_masked": sum(d["masked_think"] > 0 for d in data if d["thinking"] == "off"),
                "off_think_blocks_masked": sum(d["masked_think"] for d in data if d["thinking"] == "off"),
                "default_think_open_masked": sum(d["masked_think"] for d in data if d["thinking"] != "off"),
                "empty_think_ids": empty_think_ids, "assistant_header_ids": assistant_header},
       "render": RENDER}


def show(d, width=1600):
    # Masked spans in [[...]], trained spans bare; enough to see the boundaries.
    out, i = [], 0
    while i < len(d["ids"]):
        j = i
        trained = d["labels"][i] != -100
        while j < len(d["ids"]) and (d["labels"][j] != -100) == trained:
            j += 1
        s = tok.decode(d["ids"][i:j])
        out.append(s if trained else "[[" + s + "]]")
        i = j
    return "".join(out)[-width:]


if RENDER in ("think", "trace") or DRY:
    res["parity"] = parity(plain, data[:len(plain)])
if RENDER == "trace":
    res["trace_parity"] = trace_parity
    res["data"]["trace_sequences"] = trace_parity["turns"]
if DRY:
    print(json.dumps(res["data"], indent=1))
    for mode in ("default", "off") + (("trace",) if RENDER == "trace" else ()):
        d = next(d for d in data if d["thinking"] == mode and not d["truncated"])
        print(f"\n===== thinking={mode} (last 1600 chars; [[masked]])\n{show(d)}")
    # Trap fixture: the other render's guard too. G6's must FAIL (its default records train
    # from "assistant\n"), the think render's must PASS.
    if RENDER == "trace":
        print(f"\n===== trace guard (vLLM prompt token counts): {'PASS' if trace_parity['pass'] else 'FAIL'}\n"
              f"{json.dumps(trace_parity, indent=1)}")
    this = "think" if RENDER == "trace" else RENDER
    other = "g6" if this == "think" else "think"
    for how, st in ((this, res["parity"]), (other, parity(plain, [encode(r, other) for r in plain]))):
        print(f"\n===== parity guard, RENDER={how}{' (this run)' if how == RENDER else ''}: "
              f"{'PASS' if st['pass'] else 'FAIL'}\n{json.dumps(st, indent=1)}")
    sys.exit(0)
if RENDER in ("think", "trace") and not res["parity"]["pass"]:
    sys.exit(f"parity guard FAILED, not training:\n{json.dumps(res['parity'], indent=1)}")
if RENDER == "trace" and not trace_parity["pass"]:
    sys.exit(f"trace guard FAILED, not training:\n{json.dumps(trace_parity, indent=1)}")

lean_scan.install()
chunked_ce.install(int(os.environ.get("CE_CHUNK", "256")))
if GUARD == "hw":
    route1.ABORT_REASONS = ("hw_thermal", "hw_power_brake")
    route1.ABORT_TEMP_C = 90
elif GUARD != "g5":
    sys.exit(f"GUARD must be g5 or hw, not {GUARD}")

import bitsandbytes as bnb  # noqa: E402
import peft  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

sampler = route1.Sampler(interval=0.5)
sampler.start()
torch.cuda.init()
t_start = time.time()
res["platform_profile"] = os.environ.get("PLATFORM_PROFILE", "not passed")
res["gpu"] = torch.cuda.get_device_name()
res["abort_rule"] = {"reasons": list(route1.ABORT_REASONS), "temp_C": route1.ABORT_TEMP_C}
res["steps"] = []


def dump():
    res["time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    res["wall_s"] = round(time.time() - t_start, 1)
    res["phase_peaks"] = sampler.report()
    res["abort"] = sampler.abort
    res["chunked_ce"] = chunked_ce.report()
    with open(f"/out/{label}.json", "w") as f:
        json.dump(res, f, indent=1)


def save(path):
    model.save_pretrained(path)
    tok.save_pretrained(path)
    for f in os.listdir(path):  # safetensors writes 0600 (gpu-lab qlora.py does the same);
        p = os.path.join(path, f)  # files only: the final save's listing holds checkpoint dirs
        if os.path.isfile(p):
            os.chmod(p, 0o644)


sampler.phase = "load"
model, info, load_s = route1.load()
res["load"] = {"seconds": round(load_s, 1), "hook": route1.hook_summary(),
               "missing_keys": len(info["missing_keys"]), "mamba": route1.mamba_kernels()}
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
torch.manual_seed(SEED)
model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.05, bias="none",
                                         task_type="CAUSAL_LM", target_modules=TARGETS))
model.peft_config["default"].base_model_name_or_path = REPO
params = [p for _, p in model.named_parameters() if p.requires_grad]
res["lora"] = {"trainable_params": sum(p.numel() for p in params),
               "n_modules": sum(isinstance(m, peft.tuners.lora.LoraLayer) for m in model.modules())}
opt = bnb.optim.PagedAdamW8bit(params, lr=LR, weight_decay=0.0)
steps_per_epoch = math.ceil(len(data) / PER_STEP)
total_steps = steps_per_epoch * EPOCHS
warmup = max(1, round(total_steps * 0.03))
sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
res["schedule"] = {"steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "warmup_steps": warmup}
model.train()
gc.collect()
torch.cuda.empty_cache()
dump()
print(f"loaded in {load_s:.0f}s; {res['lora']}; {res['schedule']}", flush=True)

step = 0
for epoch in range(EPOCHS):
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(SEED + epoch)).tolist()
    for s in range(steps_per_epoch):
        if sampler.abort:
            break
        batch = order[s * PER_STEP:(s + 1) * PER_STEP]
        n_step = sum(n_items[k] for k in batch)
        sampler.phase = "train"
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        loss_step = 0.0
        for k in batch:
            if n_items[k] == 0:
                continue
            x = torch.tensor([data[k]["ids"]], device=0)
            y = torch.tensor([data[k]["labels"]], device=0)
            out = model(input_ids=x, labels=y, use_cache=False)
            loss = out.loss * (n_items[k] / n_step)  # token-mean over the whole step
            loss.backward()
            loss_step += loss.item()
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        dt = time.time() - t0
        toks = sum(len(data[k]["ids"]) for k in batch)
        res["steps"].append({"step": step, "epoch": epoch, "loss": round(loss_step, 4),
                             "grad_norm": round(gnorm, 4), "lr": sched.get_last_lr()[0],
                             "seconds": round(dt, 2), "tokens": toks,
                             "torch_peak_GiB": gib(torch.cuda.max_memory_allocated())})
        if step % 5 == 0 or step == 1:
            print(f"step {step}/{total_steps} loss {loss_step:.4f} gnorm {gnorm:.3f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {dt:.1f}s {toks / dt:.0f} tok/s", flush=True)
        dump()
    if sampler.abort:
        break
    save(f"{OUT}/checkpoint-{step}")

sampler.phase = "save"
if sampler.abort:
    save(f"{OUT}-partial")
    print(f"ABORTED: {sampler.abort} after step {step}", flush=True)
else:
    save(OUT)
res["final_step"] = step
losses = [r["loss"] for r in res["steps"]]
if losses:
    res["loss_first"], res["loss_last"] = losses[0], losses[-1]
sampler.stop = True
dump()
print(f"done: {step} steps, loss {res.get('loss_first')} -> {res.get('loss_last')}, "
      f"abort={sampler.abort}", flush=True)

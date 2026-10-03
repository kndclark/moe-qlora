"""Q1 (docs/next-model-plan.md): does Qwen3.8-27B QLoRA-train on one 24 GB card?

G4's idle load, then G5's training steps, on Qwen/Qwen3.8-27B @ 1d4bf0f2 (BF16 weights,
quantized to NF4 on load). Every choice below is the pre-registered one; see the doc.

Load: Qwen3_5ForCausalLM (text only) from the checkpoint's text_config. transformers 5.16.1
maps model.language_model.* onto it (conversion_mapping.py, PrefixChange for qwen3_5_text)
and ignores ^model.visual.* and ^mtp.*, so the vision tower and MTP head are never built.
NF4, double quant, bf16 compute (qlora.py); embeddings and lm_head stay bf16.
caching_allocator_warmup is disabled, as route1.load() does: it only pre-reserves memory.

Train: LoRA r16 / alpha 32 / dropout 0.05 on G6q's targets carried over (TARGETS), gradient
checkpointing (non-reentrant) + enable_input_require_grads, no fp32 upcast, paged AdamW 8-bit
(lr 1e-4 constant, wd 0, clip 1.0), batch 1, chunked cross-entropy (chunk 256, G6q's F1; the
loss is chunked_ce.chunked_loss under a Qwen3_5ForCausalLM forward defined here).
Lengths LENGTHS (512,1024,2048), WARMUP 2 + STEPS 10 each; an OOM ends the sweep.

Data: G6q's records rendered with Qwen3.8's template (thinking-on records at
reasoning_effort low, "<think>\\n" masked, "\\n</think>\\n\\n" + content trained; "off" records
with "<think>\\n\\n</think>\\n\\n" masked), concatenated in a seed-0 order and cut into exact
L-token windows. A thinking-on record is tokenized in pieces split after each
"<|im_start|>assistant\\n<think>\\n", so that boundary is a token boundary, as the server's
prompt ends there; parity() checks every trained turn's boundary against the server prompt.

DRY_RUN=1: data only (token counts at low and xhigh effort, parity), no GPU; run it without
--gpus. Otherwise run on the laptop:
  GUARD=hw probes/gpurun.sh q1-laptop /probes/q1_probe.py q1-laptop
Output: /out/<label>.json, rewritten after every step.
"""
import gc
import json
import os
import statistics
import sys
import time

import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, "/gpulab/training")
from tools import TOOLS  # noqa: E402
from transformers import AutoConfig, AutoTokenizer  # noqa: E402

label = sys.argv[1]
REPO, REV = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DATASET = os.environ.get("DATASET") or "/out/research_dataset_g6q.json"
DRY = os.environ.get("DRY_RUN") == "1"
GUARD = os.environ.get("GUARD", "hw")
LENGTHS = [int(x) for x in os.environ.get("LENGTHS", "512,1024,2048").split(",")]
WARMUP, STEPS = 2, 10
CE_CHUNK = int(os.environ.get("CE_CHUNK", "256"))
RANK, LR, SEED, EFFORT = 16, 1e-4, 0, "low"
TARGETS = (r".*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
           r"|.*\.linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a)$"
           r"|.*\.mlp\.(up_proj|down_proj)$")
GiB = 2**30


def gib(n):
    return round(n / GiB, 3)


res = {"label": label, "model": REPO, "revision": REV, "dataset": DATASET, "guard": GUARD,
       "config": {"lengths": LENGTHS, "warmup": WARMUP, "steps": STEPS, "rank": RANK, "alpha": 2 * RANK,
                  "dropout": 0.05, "lr": LR, "weight_decay": 0.0, "clip": 1.0, "optimizer": "paged_adamw_8bit",
                  "batch": 1, "targets": TARGETS, "ce_chunk": CE_CHUNK, "reasoning_effort": EFFORT,
                  "seed": SEED, "fp32_upcast": False, "gradient_checkpointing": "non-reentrant"}}


def dump():
    res["time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(f"/out/{label}.json", "w") as f:
        json.dump(res, f, indent=1)


# ---------------------------------------------------------------- data
tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
records = json.load(open(DATASET))
HEADER = "<|im_start|>assistant\n"
THINK_OPEN = "<think>\n"
OFF_THINK = "<think>\n\n</think>\n\n"
header_ids = tok.encode(HEADER, add_special_tokens=False)
think_open_ids = tok.encode(THINK_OPEN, add_special_tokens=False)
off_think_ids = tok.encode(OFF_THINK, add_special_tokens=False)
im_end = tok.convert_tokens_to_ids("<|im_end|>")


def lightning_messages(msgs):
    # g6_train.py: the XML tool-call templates want tool-call arguments as objects.
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
    return TOOLS + record.get("extra_tools", [])


def kwargs_for(record, effort):
    if record.get("thinking") == "off":
        return {"enable_thinking": False}
    return {"enable_thinking": True, "reasoning_effort": effort}


def encode(record, effort):
    off = record.get("thinking") == "off"
    text = tok.apply_chat_template(lightning_messages(record["messages"]), tools=tools_for(record),
                                   tokenize=False, **kwargs_for(record, effort))
    if off:
        pieces = [text]
    else:  # history turns read "<think>\n\n</think>\n\n"; split so "<think>\n" ends a piece
        marker = HEADER + THINK_OPEN
        parts = text.split(marker)
        pieces = [p + marker for p in parts[:-1]] + [parts[-1]]
    ids = []
    for p in pieces:
        if p:
            ids += tok(p, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(ids)
    pre = off_think_ids if off else think_open_ids
    turns, masked, i = [], 0, 0
    while i < len(ids):
        if ids[i:i + len(header_ids)] == header_ids:
            start = i + len(header_ids)
            if ids[start:start + len(pre)] == pre:
                start += len(pre)
                masked += 1
            turns.append((i, start))
            end = start
            while end < len(ids) and ids[end] != im_end:
                end += 1
            end = min(end + 1, len(ids))
            for j in range(start, end):
                labels[j] = ids[j]
            i = end
        else:
            i += 1
    return {"ids": ids, "labels": labels, "turns": turns, "masked": masked, "off": off}


def parity(recs, encs, effort):
    """g6_train.py's guard: every trained turn's header-to-first-trained-token span must equal
    the eval server's prompt for that turn (history as the harness sends it, no reasoning)."""
    st = {"turns": 0, "boundary_ok": 0, "boundary_bad": 0, "exact": 0, "history_diff_turns": 0,
          "bad_examples": []}
    for rec, d in zip(recs, encs):
        msgs = lightning_messages(rec["messages"])
        ks = [j for j, m in enumerate(msgs) if m["role"] == "assistant"]
        for k, (h, start) in zip(ks, d["turns"]):
            st["turns"] += 1
            srv = tok(tok.apply_chat_template(msgs[:k], tools=tools_for(rec), tokenize=False,
                                              add_generation_prompt=True, **kwargs_for(rec, effort)),
                      add_special_tokens=False)["input_ids"]
            sh = max(j for j in range(len(srv)) if srv[j:j + len(header_ids)] == header_ids)
            if srv[sh:] == d["ids"][h:start]:
                st["boundary_ok"] += 1
            else:
                st["boundary_bad"] += 1
                if len(st["bad_examples"]) < 3:
                    st["bad_examples"].append({"turn": k, "server": tok.decode(srv[sh:]),
                                               "train": tok.decode(d["ids"][h:start])})
            if srv == d["ids"][:start]:
                st["exact"] += 1
            elif srv[:sh] != d["ids"][:h]:
                st["history_diff_turns"] += 1
    st["pass"] = st["boundary_bad"] == 0 and st["boundary_ok"] > 0
    return st


def counts(encs):
    lens = sorted(len(d["ids"]) for d in encs)
    n = len(lens)

    def pct(q):
        return lens[min(n - 1, int(q * n))]
    return {"records": n, "tokens": sum(lens), "assistant_tokens": sum(sum(l != -100 for l in d["labels"]) for d in encs),
            "p50": pct(0.5), "p90": pct(0.9), "p95": pct(0.95), "max": lens[-1],
            "over_512": sum(x > 512 for x in lens), "over_1024": sum(x > 1024 for x in lens),
            "over_2048": sum(x > 2048 for x in lens),
            "no_trained_tokens": sum(all(l == -100 for l in d["labels"]) for d in encs),
            "off_records": sum(d["off"] for d in encs),
            "turns_with_think_masked": sum(d["masked"] for d in encs)}


t0 = time.time()
enc = {e: [encode(r, e) for r in records] for e in ("low", "xhigh")}
res["tokens"] = {e: counts(enc[e]) for e in enc}
res["tokens"]["render_seconds"] = round(time.time() - t0, 1)
res["tokens"]["ids"] = {"header": header_ids, "think_open": think_open_ids, "off_think": off_think_ids,
                        "im_end": im_end}
res["parity"] = parity(records, enc[EFFORT], EFFORT)
data = enc[EFFORT]
order = torch.randperm(len(data), generator=torch.Generator().manual_seed(SEED)).tolist()
stream_ids = [t for k in order for t in data[k]["ids"]]
stream_lab = [t for k in order for t in data[k]["labels"]]
res["stream_tokens"] = len(stream_ids)


def windows(L, n):
    out, skipped, k = [], 0, 0
    while len(out) < n and (k + 1) * L <= len(stream_ids):
        ids, lab = stream_ids[k * L:(k + 1) * L], stream_lab[k * L:(k + 1) * L]
        if sum(l != -100 for l in lab[1:]) == 0:
            skipped += 1
        else:
            out.append((k, ids, lab))
        k += 1
    return out, skipped


print(json.dumps({"tokens": res["tokens"], "parity": {k: v for k, v in res["parity"].items()}}, indent=1), flush=True)
dump()
if DRY:
    d = next(x for x in data if not x["off"])
    print("\n===== a thinking-on record, last 1200 chars ([[masked]]):")
    out, i = [], 0
    while i < len(d["ids"]):
        j, tr = i, d["labels"][i] != -100
        while j < len(d["ids"]) and (d["labels"][j] != -100) == tr:
            j += 1
        s = tok.decode(d["ids"][i:j])
        out.append(s if tr else "[[" + s + "]]")
        i = j
    print("".join(out)[-1200:])
    sys.exit(0)
if not res["parity"]["pass"]:
    sys.exit(f"parity guard FAILED, not training:\n{json.dumps(res['parity'], indent=1)}")

# ---------------------------------------------------------------- GPU
import bitsandbytes as bnb  # noqa: E402
import chunked_ce  # noqa: E402  (its nemotron_h install is not used; only chunked_loss)
import peft  # noqa: E402
import route1  # noqa: E402  (Sampler: NVML peaks, thermal guard)
import transformers.modeling_utils as mu  # noqa: E402
import transformers.models.qwen3_5.modeling_qwen3_5 as q35  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import BitsAndBytesConfig  # noqa: E402
from transformers.modeling_outputs import CausalLMOutputWithPast  # noqa: E402

if GUARD == "hw":
    route1.ABORT_REASONS = ("hw_thermal", "hw_power_brake")
    route1.ABORT_TEMP_C = 90
elif GUARD != "g5":
    sys.exit(f"GUARD must be g5 or hw, not {GUARD}")
pynvml, nvh = route1.pynvml, route1.nvh
nvml_total = pynvml.nvmlDeviceGetMemoryInfo(nvh).total
PASS_LINE = nvml_total - 0.5 * GiB
res["gpu_env"] = {"nvml_total_GiB": gib(nvml_total), "pass_line_GiB": gib(PASS_LINE),
                  "nvml_used_before_cuda_GiB": gib(pynvml.nvmlDeviceGetMemoryInfo(nvh).used),
                  "platform_profile": os.environ.get("PLATFORM_PROFILE", "not passed"),
                  "abort_rule": {"reasons": list(route1.ABORT_REASONS), "temp_C": route1.ABORT_TEMP_C}}
try:
    res["gpu_env"]["enforced_power_limit_W"] = pynvml.nvmlDeviceGetEnforcedPowerLimit(nvh) / 1000
except pynvml.NVMLError as e:
    res["gpu_env"]["enforced_power_limit_W"] = str(e)

# Kernel path: count calls to the functions the DeltaNet forward looks up at call time.
calls = {}


def counted(name, fn):
    def wrapper(*a, **k):
        calls[name] = calls.get(name, 0) + 1
        return fn(*a, **k)
    return wrapper


kpath = {}
for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
             "causal_conv1d_update"):
    fn = getattr(q35, name)
    kpath[name] = {"repr": repr(fn)[:200], "module": getattr(fn, "__module__", None),
                   "qualname": getattr(fn, "__qualname__", None)}
    setattr(q35, name, counted(name, fn))
for pkg in ("fla", "causal_conv1d", "flash_attn", "kernels"):
    try:
        __import__(pkg)
        kpath[pkg] = "importable"
    except Exception as e:
        kpath[pkg] = type(e).__name__
res["kernel_path"] = kpath

_orig_forward = q35.Qwen3_5ForCausalLM.forward


def chunked_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                    inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
    if labels is None or not torch.is_grad_enabled():
        return _orig_forward(self, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                             past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                             use_cache=use_cache, logits_to_keep=logits_to_keep, **kwargs)
    out = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                     past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
    calls["chunked_ce"] = calls.get("chunked_ce", 0) + 1
    loss = chunked_ce.chunked_loss(self.lm_head, out.last_hidden_state, labels, CE_CHUNK)
    return CausalLMOutputWithPast(loss=loss, logits=None)


q35.Qwen3_5ForCausalLM.forward = chunked_forward

sampler = route1.Sampler(interval=0.1)
sampler.start()


def phase_peak(ph):
    p = sampler.peaks.get(ph, {})
    return {"nvml_peak_GiB": gib(p.get("nvml_used", 0)), "temp_C": p.get("temp_C"),
            "watts": round(p.get("watts", 0.0), 1), "rss_GiB": gib(p.get("rss", 0)),
            "reasons": {k: v for k, v in p.get("reasons", {}).items() if v}}


def mem(tag):
    torch.cuda.synchronize()
    return {"tag": tag, "torch_allocated_GiB": gib(torch.cuda.memory_allocated()),
            "torch_reserved_GiB": gib(torch.cuda.memory_reserved()),
            "nvml_used_GiB": gib(pynvml.nvmlDeviceGetMemoryInfo(nvh).used)}


def free():
    gc.collect()
    torch.cuda.empty_cache()


exit_code = 0
try:
    # ------------------------------------------------------------ load (G4 style)
    sampler.phase = "load"
    torch.cuda.init()
    res["gpu"] = torch.cuda.get_device_name()
    idx_path = os.path.join(os.path.dirname(
        __import__("huggingface_hub").hf_hub_download(REPO, "config.json", revision=REV)), "model.safetensors.index.json")
    keys = list(json.load(open(idx_path))["weight_map"])
    pref = {}
    for k in keys:
        p = ("model.language_model" if k.startswith("model.language_model.") else
             "model.visual" if k.startswith("model.visual.") else
             "mtp" if k.startswith("mtp.") else k.split(".")[0] if "." in k else k)
        pref[p] = pref.get(p, 0) + 1
    res["index_keys_by_prefix"] = pref
    cfg = AutoConfig.from_pretrained(REPO, revision=REV)
    tcfg = cfg.text_config
    mu.caching_allocator_warmup = lambda *a, **k: None
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model, info = q35.Qwen3_5ForCausalLM.from_pretrained(
        REPO, revision=REV, config=tcfg, quantization_config=bnb_cfg, dtype=torch.bfloat16,
        device_map={"": 0}, output_loading_info=True)
    load_s = time.time() - t0
    lin4 = [n for n, m in model.named_modules() if isinstance(m, bnb.nn.Linear4bit)]
    lin16 = [n for n, m in model.named_modules() if type(m) is torch.nn.Linear]
    packed = sum(p.numel() * p.element_size() for p in model.parameters() if p.dtype == torch.uint8)
    rest = {}
    for n, p in model.named_parameters():
        if p.dtype != torch.uint8:
            rest[str(p.dtype)] = rest.get(str(p.dtype), 0) + p.numel() * p.element_size()
    devices = sorted({str(p.device) for p in model.parameters()})
    res["load"] = {"seconds": round(load_s, 1), "class": type(model).__name__,
                   "missing_keys": sorted(info["missing_keys"]), "unexpected_keys": sorted(info["unexpected_keys"])[:20],
                   "n_unexpected": len(info["unexpected_keys"]),
                   "mismatched_keys": [str(x) for x in info.get("mismatched_keys", [])][:20],
                   "n_linear4bit": len(lin4), "bf16_linear_modules": lin16,
                   "visual_modules": sum("visual" in n for n, _ in model.named_modules()),
                   "mtp_modules": sum(n.startswith("mtp") for n, _ in model.named_modules()),
                   "param_devices": devices,
                   "attn_implementation": model.config._attn_implementation,
                   "torch_peak_allocated_GiB": gib(torch.cuda.max_memory_allocated()),
                   "torch_peak_reserved_GiB": gib(torch.cuda.max_memory_reserved()),
                   "resident": {"nf4_packed_GiB": gib(packed),
                                "other_params_GiB": {k: gib(v) for k, v in rest.items()},
                                "allocated_minus_params_GiB": gib(torch.cuda.memory_allocated() - packed
                                                                  - sum(rest.values()))},
                   "arithmetic_resident_GiB": 16.44}
    res["load"]["phase_peaks"] = phase_peak("load")
    free()
    res["idle"] = [mem("after load")]
    dump()
    print(f"loaded in {load_s:.0f}s: {res['idle'][0]}", flush=True)
    if info["missing_keys"] or devices != ["cuda:0"]:  # G4's rule: no missing keys, all on the card
        print(f"STOP: missing keys {sorted(info['missing_keys'])[:10]} / devices {devices}", flush=True)
        raise SystemExit(4)

    # sanity forward
    sampler.phase = "sanity"
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        x = tok("The capital of France is", return_tensors="pt", add_special_tokens=False).input_ids.to(0)
        logits = model(input_ids=x, use_cache=False).logits[0, -1].float()
    top = torch.topk(logits, 5).indices.tolist()
    res["sanity"] = {"finite": bool(torch.isfinite(logits).all()), "top5": [tok.decode([t]) for t in top],
                     "paris_top1": tok.decode([top[0]]) == " Paris",
                     "torch_peak_allocated_GiB": gib(torch.cuda.max_memory_allocated()),
                     "calls": dict(calls)}
    del logits
    free()
    dump()
    print(f"sanity: {res['sanity']}", flush=True)

    # ------------------------------------------------------------ LoRA (no fp32 upcast)
    sampler.phase = "peft"
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    torch.manual_seed(SEED)
    model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules=TARGETS))
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    lora_mods = [n for n, m in model.named_modules() if isinstance(m, peft.tuners.lora.LoraLayer)]
    kinds = {}
    for n in lora_mods:
        kinds[n.rsplit(".", 2)[-2] + "." + n.rsplit(".", 1)[-1]] = kinds.get(n.rsplit(".", 2)[-2] + "." + n.rsplit(".", 1)[-1], 0) + 1
    res["lora"] = {"trainable_params": sum(p.numel() for p in params), "n_modules": len(lora_mods),
                   "by_kind": kinds, "dtypes": sorted({str(p.dtype) for p in params}),
                   "non_lora_trainable": [n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n][:10]}
    opt = bnb.optim.PagedAdamW8bit(params, lr=LR, weight_decay=0.0)
    model.train()
    free()
    res["idle"].append(mem("LoRA attached"))
    dump()
    print(f"lora: {res['lora']['trainable_params']} params in {res['lora']['n_modules']} modules; "
          f"{res['idle'][-1]}", flush=True)

    # ------------------------------------------------------------ steps (G5 style)
    res["runs"] = {}
    for L in LENGTHS:
        run = {"steps": []}
        res["runs"][str(L)] = run
        wins, skipped = windows(L, WARMUP + STEPS)
        run["windows_skipped_no_labels"] = skipped
        for k, (w, ids, lab) in enumerate(wins):
            if sampler.abort:
                break
            phase = f"seq{L}_step{k}"
            free()
            non_torch = pynvml.nvmlDeviceGetMemoryInfo(nvh).used - torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            c0 = dict(calls)
            sampler.phase = phase
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                xx = torch.tensor([ids], device=0)
                yy = torch.tensor([lab], device=0)
                out = model(input_ids=xx, labels=yy, use_cache=False)
                out.loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
                opt.step()
                opt.zero_grad(set_to_none=True)
                loss = out.loss.item()
                del out, xx, yy
            except torch.OutOfMemoryError as e:
                run["oom"] = {"step": k, "message": str(e)[:800],
                              "torch_peak_allocated_GiB": gib(torch.cuda.max_memory_allocated()),
                              "torch_peak_reserved_GiB": gib(torch.cuda.max_memory_reserved()),
                              "nvml_sampled_peak_GiB": gib(sampler.peaks.get(phase, {}).get("nvml_used", 0))}
                print(f"OOM at seq {L} step {k}: {str(e)[:300]}", flush=True)
                break
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            sampler.sample()
            p = sampler.peaks[phase]
            derived = non_torch + torch.cuda.max_memory_reserved()
            run["steps"].append({
                "k": k, "window": w, "warmup": k < WARMUP, "loss": round(loss, 4), "grad_norm": round(gnorm, 4),
                "trained_tokens": sum(l != -100 for l in lab[1:]), "seconds": round(dt, 3),
                "tokens_per_s": round(L / dt, 1),
                "torch_peak_allocated_GiB": gib(torch.cuda.max_memory_allocated()),
                "torch_peak_reserved_GiB": gib(torch.cuda.max_memory_reserved()),
                "nvml_sampled_peak_GiB": gib(p["nvml_used"]), "nvml_derived_peak_GiB": gib(derived),
                "temp_C": p["temp_C"], "watts": round(p["watts"], 1),
                "reasons": {k2: v for k2, v in p["reasons"].items() if v},
                "calls": {n: calls.get(n, 0) - c0.get(n, 0) for n in calls if calls.get(n, 0) != c0.get(n, 0)}})
            print(f"seq {L} step {k}: loss {loss:.4f} {dt:.2f}s "
                  f"alloc {run['steps'][-1]['torch_peak_allocated_GiB']} "
                  f"dev {max(run['steps'][-1]['nvml_sampled_peak_GiB'], run['steps'][-1]['nvml_derived_peak_GiB'])} "
                  f"{p['temp_C']}C {p['watts']:.0f}W", flush=True)
            dump()
        st = run["steps"]
        meas = [s for s in st if not s["warmup"]]
        if st:
            dev = [max(s["nvml_sampled_peak_GiB"], s["nvml_derived_peak_GiB"]) for s in st]
            run["summary"] = {
                "steps_done": len(st), "measured_steps": len(meas),
                "all_losses_finite": all(s["loss"] == s["loss"] and abs(s["loss"]) != float("inf") for s in st),
                "loss_first": st[0]["loss"], "loss_last": st[-1]["loss"],
                "median_seconds": round(statistics.median(s["seconds"] for s in meas), 3) if meas else None,
                "median_tokens_per_s": round(statistics.median(s["tokens_per_s"] for s in meas), 1) if meas else None,
                "torch_peak_allocated_GiB": max(s["torch_peak_allocated_GiB"] for s in st),
                "torch_peak_reserved_GiB": max(s["torch_peak_reserved_GiB"] for s in st),
                "nvml_sampled_peak_GiB": max(s["nvml_sampled_peak_GiB"] for s in st),
                "device_peak_GiB": max(dev),
                "device_peak_measured_only_GiB": max(dev[WARMUP:]) if len(dev) > WARMUP else None,
                "max_temp_C": max(s["temp_C"] for s in st), "max_watts": max(s["watts"] for s in st),
                "reason_samples": {r: sum(s["reasons"].get(r, 0) for s in st)
                                   for r in {r for s in st for r in s["reasons"]}}}
        ok = (len(st) == WARMUP + STEPS and "oom" not in run and not sampler.abort
              and run.get("summary", {}).get("all_losses_finite"))
        run["verdict"] = ("OOM" if "oom" in run else f"ABORT: {sampler.abort}" if sampler.abort
                          else "PASS" if ok and run["summary"]["device_peak_GiB"] <= gib(PASS_LINE)
                          else "RUNS, OVER THE LINE" if ok else "INCOMPLETE")
        print(f"seq {L}: {run['verdict']} {run.get('summary')}", flush=True)
        dump()
        if "oom" in run or sampler.abort:
            break
except torch.OutOfMemoryError as e:
    res["oom_outside_steps"] = {"phase": sampler.phase, "message": str(e)[:800],
                                "torch_peak_allocated_GiB": gib(torch.cuda.max_memory_allocated())}
    print(f"OOM in phase {sampler.phase}: {str(e)[:300]}", flush=True)
    exit_code = 3
finally:
    res["kernel_calls_total"] = dict(calls)
    res["phase_peaks"] = sampler.report()
    res["abort"] = sampler.abort
    sampler.stop = True
    dump()
sys.exit(exit_code)

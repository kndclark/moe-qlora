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
loss is chunked_ce.chunked_loss under a Qwen3_5ForCausalLM forward).
The render, load, targets and loss live in q38.py, shared with q2_train.py.
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
import q38  # noqa: E402  (render, load and loss, shared with q2_train.py)
from q38 import EFFORT, REPO, REV, TARGETS, counts, gib  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

label = sys.argv[1]
DATASET = os.environ.get("DATASET") or "/out/research_dataset_g6q.json"
DRY = os.environ.get("DRY_RUN") == "1"
GUARD = os.environ.get("GUARD", "hw")
LENGTHS = [int(x) for x in os.environ.get("LENGTHS", "512,1024,2048").split(",")]
WARMUP, STEPS = 2, 10
CE_CHUNK = int(os.environ.get("CE_CHUNK", "256"))
RANK, LR, SEED = 16, 1e-4, 0
GiB = q38.GiB


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
render = q38.Render(tok)
header_ids, think_open_ids, off_think_ids, im_end = (render.header_ids, render.think_open_ids,
                                                     render.off_think_ids, render.im_end)
encode, parity = render.encode, render.parity


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
import peft  # noqa: E402
import route1  # noqa: E402  (Sampler: NVML peaks, thermal guard)
from peft import LoraConfig, get_peft_model  # noqa: E402

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
res["kernel_path"] = q38.count_kernel_calls(calls)
q38.install_chunked_ce(CE_CHUNK, calls)

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
    res["index_keys_by_prefix"] = q38.index_prefixes()
    torch.cuda.reset_peak_memory_stats()
    model, info, load_s = q38.load()
    res["load"] = {"seconds": round(load_s, 1), **q38.inventory(model, info),
                   "torch_peak_allocated_GiB": gib(torch.cuda.max_memory_allocated()),
                   "torch_peak_reserved_GiB": gib(torch.cuda.max_memory_reserved()),
                   "arithmetic_resident_GiB": 16.44}
    devices = res["load"]["param_devices"]
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

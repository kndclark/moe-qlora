"""G5: LoRA training steps on Route 1, placement A (plan.md G5).

Setup (from plan.md): LoRA r=16, alpha=32, dropout 0 on attention q/k/v/o_proj,
mamba in_proj and shared_experts.up_proj/down_proj; gradient checkpointing,
non-reentrant; enable_input_require_grads; no fp32 upcast; bnb PagedAdamW8bit
(lr 1e-4 as qlora.py's default, weight decay 0 as Trainer's); batch 1.

Data: research_dataset_v3.json, shuffled with qlora.py's dataset.build rule
(random.Random(0)), each record rendered with Lightning's own chat template and
the lab's TOOLS. The template iterates tool-call arguments as a mapping, so the
records' JSON-string arguments are parsed first. Rendered records are packed
end to end and cut into sequences of exactly L tokens, labels = inputs, so every
step at length L carries L real tokens.

Per length L (512, 1024, 2048, in that order): 3 warm-up and 20 measured steps.
Each step records loss, seconds, tokens/s, torch peak allocated and reserved,
and two device-wide numbers: the NVML peak sampled every 0.1 s, and a derived
peak = (NVML used - torch reserved) just before the step + the step's torch
peak reserved, which a short spike cannot slip past. The gate uses the larger.
An OOM is recorded as the result for that L and ends the sweep (a longer L
cannot fit). The thermal guard (route1.Sampler) stops the run between steps.

Then, at the longest L that fit (1024 preferred), where the time goes:
  - per block type, forward + recompute + backward of every block, each timed
    in isolation with CUDA events on real hidden states (what one training
    step spends in mamba, attention and MoE blocks);
  - torch.profiler over one full step: the top CUDA ops by total time.

Epoch-time arithmetic for the gate: tokens in one epoch (all 950 records,
rendered, untruncated; packing trains every token) / median tokens/s at 1024.

Run (laptop, platform profile max-power, other GPU apps closed):
  docker run --rm --init --gpus all --ipc=host -v /srv/model-cache:/hf:ro \
    -v ~/moe-qlora/probes:/probes:ro -v ~/moe-qlora/results:/out \
    -v ~/gpu-lab/training:/gpulab/training:ro \
    -e PLATFORM_PROFILE=$(cat /sys/firmware/acpi/platform_profile) \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e PYTHONDONTWRITEBYTECODE=1 --user $(id -u):$(id -g) \
    --entrypoint python3 gpu-lab:training /probes/g5_train_step.py LABEL [512,1024,2048]
"""
import gc
import json
import os
import random
import statistics
import sys
import time

import bitsandbytes as bnb
import peft
import torch
import transformers
import transformers.models.nemotron_h.modeling_nemotron_h as nh
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route1  # noqa: E402
from route1 import REPO, REV, gib  # noqa: E402

sys.path.insert(0, "/gpulab/training")
from tools import TOOLS  # noqa: E402

label = sys.argv[1]
SEQS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "512,1024,2048").split(",")]
WARMUP, MEASURED = 3, 20
DATASET = "/gpulab/training/research_dataset_v3.json"
TARGETS = r".*\.mixer\.(q_proj|k_proj|v_proj|o_proj|in_proj)$|.*\.mixer\.shared_experts\.(up_proj|down_proj)$"

nvml_total = route1.pynvml.nvmlDeviceGetMemoryInfo(route1.nvh).total
PASS_LINE = nvml_total - 0.5 * route1.GiB
baseline_nvml_used = route1.pynvml.nvmlDeviceGetMemoryInfo(route1.nvh).used
sampler = route1.Sampler(interval=0.1)
sampler.start()
torch.cuda.init()
t_start = time.time()
res = {"label": label, "model": REPO, "revision": REV, "seqs": SEQS,
       "platform_profile": os.environ.get("PLATFORM_PROFILE", "not passed"),
       "gpu": torch.cuda.get_device_name(), "nvml_total_GiB": gib(nvml_total), "pass_line_GiB": gib(PASS_LINE),
       "torch": torch.__version__, "transformers": transformers.__version__,
       "bitsandbytes": bnb.__version__, "peft": peft.__version__,
       "baseline_nvml_used_GiB_before_cuda": gib(baseline_nvml_used)}
try:
    res["enforced_power_limit_W"] = route1.pynvml.nvmlDeviceGetEnforcedPowerLimit(route1.nvh) / 1000
except route1.pynvml.NVMLError as e:
    res["enforced_power_limit_W"] = str(e)


def dump():
    res["time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    res["wall_s"] = round(time.time() - t_start, 1)
    res["phase_peaks"] = sampler.report()
    res["abort"] = sampler.abort
    with open(f"/out/{label}.json", "w") as f:
        json.dump(res, f, indent=1)


def free():
    gc.collect()
    torch.cuda.empty_cache()


# ---- data
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


stream, lengths, render_errors = [], [], []
for i, r in enumerate(records):
    try:
        text = tok.apply_chat_template(lightning_messages(r["messages"]), tools=TOOLS, tokenize=False)
    except Exception as e:
        render_errors.append([i, type(e).__name__, str(e)[:200]])
        continue
    ids = tok(text, add_special_tokens=False)["input_ids"]
    lengths.append(len(ids))
    stream.extend(ids)
lengths_sorted = sorted(lengths)
res["data"] = {"dataset": DATASET, "records": len(records), "rendered": len(lengths),
               "render_errors": render_errors[:5], "n_render_errors": len(render_errors),
               "epoch_tokens": len(stream),
               "record_tokens_p50": lengths_sorted[len(lengths) // 2],
               "record_tokens_p90": lengths_sorted[int(len(lengths) * 0.9)],
               "record_tokens_max": lengths_sorted[-1],
               "epoch_tokens_if_truncated_at_1024": sum(min(n, 1024) for n in lengths),
               "sample_render_head": tok.decode(stream[:400])}
assert len(stream) >= (WARMUP + MEASURED) * max(SEQS), len(stream)

# ---- model
sampler.phase = "load"
model, info, load_s = route1.load()
res["load"] = {"seconds": round(load_s, 1), "hook": route1.hook_summary(),
               "missing_keys": len(info["missing_keys"]), "mamba": route1.mamba_kernels(),
               "experts_implementation": getattr(model.config, "_experts_implementation", None),
               "attn_implementation": model.config._attn_implementation}
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                                         task_type="CAUSAL_LM", target_modules=TARGETS))
lora_kinds = {}
for n, m in model.named_modules():
    if isinstance(m, peft.tuners.lora.LoraLayer):
        kind = n.rsplit(".", 1)[-1]
        if "shared_experts" in n:
            kind = "shared_experts." + kind
        lora_kinds[kind] = lora_kinds.get(kind, 0) + 1
trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
res["lora"] = {"modules_by_kind": lora_kinds, "n_modules": sum(lora_kinds.values()),
               "trainable_params": sum(p.numel() for _, p in trainable),
               "non_lora_trainable": [n for n, _ in trainable if "lora_" not in n][:10]}
opt = bnb.optim.PagedAdamW8bit([p for _, p in trainable], lr=1e-4, weight_decay=0.0)
model.train()
free()
sampler.phase = "idle_after_peft"
time.sleep(2)
res["idle_after_peft"] = {"torch_allocated_GiB": gib(torch.cuda.memory_allocated()),
                          "torch_reserved_GiB": gib(torch.cuda.memory_reserved()),
                          "nvml_used_GiB": gib(sampler.sample())}
dump()


def step(ids):
    x = torch.tensor([ids], device=0)
    out = model(input_ids=x, labels=x, use_cache=False)
    out.loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return out.loss.item()


# ---- sweep
res["runs"] = {}
fit = []
for L in SEQS:
    run = {"steps": []}
    res["runs"][str(L)] = run
    chunks = [stream[k * L:(k + 1) * L] for k in range(WARMUP + MEASURED)]
    for k, ids in enumerate(chunks):
        if sampler.abort:
            break
        phase = f"seq{L}_step{k}"
        free()
        non_torch = route1.pynvml.nvmlDeviceGetMemoryInfo(route1.nvh).used - torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        sampler.phase = phase
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            loss = step(ids)
        except torch.OutOfMemoryError as e:
            run["oom"] = {"step": k, "message": str(e)[:600],
                          "torch_peak_allocated_GiB": gib(torch.cuda.max_memory_allocated()),
                          "torch_peak_reserved_GiB": gib(torch.cuda.max_memory_reserved())}
            opt.zero_grad(set_to_none=True)
            break
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        sampler.sample()
        p = sampler.peaks[phase]
        derived = non_torch + torch.cuda.max_memory_reserved()
        run["steps"].append({
            "k": k, "warmup": k < WARMUP, "loss": round(loss, 4), "seconds": round(dt, 3),
            "tokens_per_s": round(L / dt, 1),
            "torch_peak_allocated_GiB": gib(torch.cuda.max_memory_allocated()),
            "torch_peak_reserved_GiB": gib(torch.cuda.max_memory_reserved()),
            "nvml_sampled_peak_GiB": gib(p["nvml_used"]), "nvml_derived_peak_GiB": gib(derived),
            "temp_C": p["temp_C"], "watts": round(p["watts"], 1),
            "reasons": {k2: v for k2, v in p["reasons"].items() if v}})
        dump()
    meas = [s for s in run["steps"] if not s["warmup"]]
    if meas:
        peaks = [s["torch_peak_allocated_GiB"] for s in meas]
        run["summary"] = {
            "measured_steps": len(meas),
            "all_losses_finite": all(s["loss"] == s["loss"] and abs(s["loss"]) != float("inf") for s in meas),
            "median_seconds": round(statistics.median(s["seconds"] for s in meas), 3),
            "median_tokens_per_s": round(statistics.median(s["tokens_per_s"] for s in meas), 1),
            "torch_peak_allocated_GiB_min_max": [min(peaks), max(peaks)],
            "torch_peak_first5_vs_last5_GiB": [round(sum(peaks[:5]) / 5, 3), round(sum(peaks[-5:]) / 5, 3)],
            "device_peak_GiB": max(max(s["nvml_sampled_peak_GiB"], s["nvml_derived_peak_GiB"]) for s in meas),
            "max_temp_C": max(s["temp_C"] for s in meas),
            "max_watts": max(s["watts"] for s in meas)}
        if len(meas) == MEASURED:
            fit.append(L)
    dump()
    if "oom" in run or sampler.abort:
        break

# ---- gate arithmetic
r1024 = res["runs"].get("1024", {}).get("summary")
if r1024 and 1024 in fit:
    hours = res["data"]["epoch_tokens"] / r1024["median_tokens_per_s"] / 3600
    res["gate"] = {"seq1024_device_peak_GiB": r1024["device_peak_GiB"], "pass_line_GiB": gib(PASS_LINE),
                   "memory_pass": r1024["device_peak_GiB"] <= gib(PASS_LINE),
                   "epoch_hours_at_seq1024": round(hours, 2), "time_pass": hours <= 8.0}
    res["gate"]["pass"] = res["gate"]["memory_pass"] and res["gate"]["time_pass"]
else:
    res["gate"] = {"pass": False, "reason": "seq 1024 did not complete 20 measured steps"}
dump()

# ---- where the time goes
L = 1024 if 1024 in fit else (max(fit) if fit else None)
if L and not sampler.abort:
    sampler.phase = f"breakdown_seq{L}"
    base = model.base_model.model
    layers = base.model.layers
    ids = torch.tensor([stream[:L]], device=0)
    pos = torch.arange(L, device=0)[None]
    try:
        inputs = {}
        hooks = [b.register_forward_pre_hook(lambda mod, a, i=i: inputs.__setitem__(i, a[0].detach()))
                 for i, b in enumerate(layers)]
        with torch.no_grad():
            base.model(input_ids=ids, use_cache=False)
        for h in hooks:
            h.remove()
        kw = {"config": base.config, "inputs_embeds": inputs[0], "attention_mask": None,
              "past_key_values": None, "position_ids": pos}
        masks = {"full_attention": nh.create_causal_mask(**kw),
                 "linear_attention": nh.create_recurrent_attention_mask(**kw)}
        per_block = []
        for i, blk in enumerate(layers):
            if sampler.abort:
                break
            times = []
            for rep in range(3):
                x = inputs[i].clone().requires_grad_(True)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                y = blk(x, attention_mask=masks.get(blk.block_type), position_ids=pos,
                        past_key_values=None, use_cache=False)
                y.backward(torch.ones_like(y))
                e.record()
                torch.cuda.synchronize()
                if rep:
                    times.append(s.elapsed_time(e) / 1000)
                del x, y
            per_block.append({"i": i, "type": blk.block_type, "seconds": round(sum(times) / len(times), 4)})
        opt.zero_grad(set_to_none=True)
        del inputs
        free()
        step_s = res["runs"][str(L)]["summary"]["median_seconds"]
        by_type = {}
        for b in per_block:
            by_type.setdefault(b["type"], 0.0)
            by_type[b["type"]] += b["seconds"]
        res["breakdown"] = {
            "seq": L, "method": "each block: checkpointed forward + recompute + backward, timed alone, CUDA events",
            "step_median_seconds": step_s,
            "seconds_by_type": {t: round(v, 3) for t, v in by_type.items()},
            "share_of_step_by_type": {t: round(v / step_s, 3) for t, v in by_type.items()},
            "rest_seconds": round(step_s - sum(by_type.values()), 3),
            "per_block": per_block}
    except torch.OutOfMemoryError as e:
        res["breakdown"] = {"oom": str(e)[:400]}
        opt.zero_grad(set_to_none=True)
        free()
    dump()

    if not sampler.abort:
        from torch.profiler import ProfilerActivity, profile
        try:
            free()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                step(stream[L:2 * L])
                torch.cuda.synchronize()
            rows = sorted(prof.key_averages(), key=lambda a: a.device_time_total, reverse=True)[:15]
            res["profiler_top_cuda_ops"] = [{"op": a.key[:90], "cuda_ms": round(a.device_time_total / 1000, 1),
                                             "calls": a.count} for a in rows]
        except Exception as e:
            res["profiler_top_cuda_ops"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        dump()

sampler.stop = True
dump()
print(json.dumps({k: res[k] for k in ("gate", "lora", "data") if k in res}, indent=1, default=str))
for L, run in res["runs"].items():
    print(L, json.dumps(run.get("summary"), default=str), json.dumps(run.get("oom"), default=str)[:300])
print("breakdown", json.dumps({k: v for k, v in res.get("breakdown", {}).items() if k != "per_block"}))

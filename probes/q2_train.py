"""Q2 (docs/next-model-plan.md): train Qwen3.8-27B on G6q's data with G6q's optimisation.

Kept from g6_train.py, as G6q ran it (docs/lightning-training.md):
  - 2 epochs; 8 records per optimizer step at batch 1, the loss normalised over every assistant
    token in the step (transformers 5.x's num_items_in_batch objective);
  - paged AdamW 8-bit, lr 1e-4, weight decay 0, cosine schedule with warmup = round(3% of
    steps), gradient-norm clip 1.0, seed 0; a fresh record order per epoch from
    torch.Generator(seed + epoch);
  - LoRA r16, alpha 32, dropout 0.05, bias none; one sequence per record, cut at MAX_LEN;
  - a checkpoint at each epoch end; the guard stops the run between optimizer steps.
Kept from q1_probe.py, the configuration that passed Q1 at 2,048 tokens (shared via q38.py):
  the NF4 load of Qwen/Qwen3.8-27B @ 1d4bf0f2 as Qwen3_5ForCausalLM, the render (thinking-on
  records at reasoning_effort low) and its parity guard, TARGETS, no fp32 upcast, gradient
  checkpointing (non-reentrant) + enable_input_require_grads, chunked cross-entropy (256).
New here:
  - a checkpoint every 52 steps as well (a sixth of the 312);
  - an OOM is caught like a guard abort: the adapter so far is saved as -partial and the run
    stops (exit 3). Neither is retried;
  - per step: torch peak allocated and reserved, NVML peak, the device peak by Q1's definition,
    temperature, power and throttle-reason samples;
  - adapter keys are saved under the checkpoint's own names, model.language_model.layers.*,
    not the text-only class's model.layers.*. vLLM v0.29.0 serves this model as
    Qwen3_5ForConditionalGeneration, maps LoRA names through its hf_to_vllm_mapper
    (model.language_model. -> language_model.model.) and looks modules up by exact name; a
    model.layers.* key matches no module and is skipped with only a DEBUG line
    (lora/model_manager.py, "No LoRA weights found for module"). To load an adapter back into
    Qwen3_5ForCausalLM with PEFT, undo the rename.

Thermal guard: route1.Sampler, as g6_train.py: GUARD=g5 (default) aborts on sw_thermal,
hw_thermal or hw_power_brake, or at 87 C; GUARD=hw on hw_thermal or hw_power_brake, or at 90 C.

Run (laptop):
  DATASET=/out/research_dataset_g6q.json GUARD=hw MAX_LEN=2048 \\
    probes/gpurun.sh q2-train /probes/q2_train.py q2-train
  DRY_RUN=1 renders and masks the data, runs the parity guard, prints the schedule and stops
  before the GPU (run it without --gpus). SMOKE_STEPS=N stops after N optimizer steps and
  saves the adapter as usual (the vLLM load test).
  DATASET unset is /out/research_dataset_g6q.json; MAX_LEN unset is 2048.
Output: /out/<label>-adapter/ (checkpoint-N every 52 steps and at each epoch end, the final
adapter at the top, tokenizer beside it), /out/<label>.json (rewritten every step).
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
import q38  # noqa: E402
from q38 import EFFORT, REPO, REV, TARGETS, gib  # noqa: E402
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup  # noqa: E402

label = sys.argv[1]
DRY = os.environ.get("DRY_RUN") == "1"
GUARD = os.environ.get("GUARD", "g5")
DATASET = os.environ.get("DATASET") or "/out/research_dataset_g6q.json"
MAX_LEN = int(os.environ.get("MAX_LEN") or 2048)
SMOKE = int(os.environ.get("SMOKE_STEPS") or 0)
CE_CHUNK = int(os.environ.get("CE_CHUNK", "256"))
EPOCHS, LR, PER_STEP, RANK, SEED, CKPT_EVERY = 2, 1e-4, 8, 16, 0, 52
OUT = f"/out/{label}-adapter"
VLLM_OLD, VLLM_NEW = "base_model.model.model.layers.", "base_model.model.model.language_model.layers."

# ---- data (Q1's render, cut at MAX_LEN)
tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
records = json.load(open(DATASET))
render = q38.Render(tok)
full = [render.encode(r, EFFORT) for r in records]
data = [{**d, "ids": d["ids"][:MAX_LEN], "labels": d["labels"][:MAX_LEN], "truncated": len(d["ids"]) > MAX_LEN}
        for d in full]
n_items = [sum(l != -100 for l in d["labels"]) for d in data]
lens = sorted(len(d["ids"]) for d in data)
steps_per_epoch = math.ceil(len(data) / PER_STEP)
total_steps = steps_per_epoch * EPOCHS
warmup = max(1, round(total_steps * 0.03))
res = {"label": label, "model": REPO, "revision": REV, "dataset": DATASET, "guard": GUARD,
       "config": {"epochs": EPOCHS, "lr": LR, "records_per_step": PER_STEP, "batch": 1, "max_len": MAX_LEN,
                  "rank": RANK, "alpha": 2 * RANK, "dropout": 0.05, "seed": SEED, "targets": TARGETS, "clip": 1.0,
                  "warmup_frac": 0.03, "weight_decay": 0.0, "optimizer": "paged_adamw_8bit", "ce_chunk": CE_CHUNK,
                  "reasoning_effort": EFFORT, "fp32_upcast": False, "gradient_checkpointing": "non-reentrant",
                  "checkpoint_every": CKPT_EVERY, "smoke_steps": SMOKE},
       "data": {"records": len(data), "tokens": sum(lens), "assistant_tokens": sum(n_items),
                "p50": lens[len(lens) // 2], "p90": lens[int(len(lens) * 0.9)], "max": lens[-1],
                "truncated": sum(d["truncated"] for d in data),
                "no_assistant_tokens": sum(n == 0 for n in n_items),
                "off_records": sum(d["off"] for d in data)},
       "q1_counts": q38.counts(full),
       "parity": render.parity(records, full, EFFORT),
       "schedule": {"steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "warmup_steps": warmup}}


def dump():
    res["time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(f"/out/{label}.json", "w") as f:
        json.dump(res, f, indent=1)


if DRY:
    print(json.dumps({k: res[k] for k in ("data", "q1_counts", "parity", "schedule")}, indent=1))
    dump()
    sys.exit(0)
if not res["parity"]["pass"]:
    sys.exit(f"parity guard FAILED, not training:\n{json.dumps(res['parity'], indent=1)}")

import bitsandbytes as bnb  # noqa: E402
import peft  # noqa: E402
import route1  # noqa: E402  (Sampler: NVML peaks, thermal guard)
from peft import LoraConfig, get_peft_model  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

if GUARD == "hw":
    route1.ABORT_REASONS = ("hw_thermal", "hw_power_brake")
    route1.ABORT_TEMP_C = 90
elif GUARD != "g5":
    sys.exit(f"GUARD must be g5 or hw, not {GUARD}")
pynvml, nvh = route1.pynvml, route1.nvh
calls = {}
res["kernel_path"] = q38.count_kernel_calls(calls)
q38.install_chunked_ce(CE_CHUNK, calls)
res["nvml_used_before_cuda_GiB"] = gib(pynvml.nvmlDeviceGetMemoryInfo(nvh).used)
sampler = route1.Sampler(interval=0.25)
sampler.start()
torch.cuda.init()
t_start = time.time()
res["platform_profile"] = os.environ.get("PLATFORM_PROFILE", "not passed")
res["gpu"] = torch.cuda.get_device_name()
res["abort_rule"] = {"reasons": list(route1.ABORT_REASONS), "temp_C": route1.ABORT_TEMP_C}
res["steps"] = []


def dump_run():
    res["wall_s"] = round(time.time() - t_start, 1)
    res["phase_peaks"] = sampler.report()
    res["abort"] = sampler.abort
    res["kernel_calls"] = dict(calls)
    dump()


def mem(tag):
    torch.cuda.synchronize()
    return {"tag": tag, "torch_allocated_GiB": gib(torch.cuda.memory_allocated()),
            "torch_reserved_GiB": gib(torch.cuda.memory_reserved()),
            "nvml_used_GiB": gib(pynvml.nvmlDeviceGetMemoryInfo(nvh).used)}


def save(path):
    model.save_pretrained(path)
    tok.save_pretrained(path)
    f = os.path.join(path, "adapter_model.safetensors")  # the vLLM names (see the docstring)
    sd = load_file(f)
    bad = [k for k in sd if not k.startswith(VLLM_OLD)]
    assert not bad, bad[:3]
    save_file({VLLM_NEW + k[len(VLLM_OLD):]: v for k, v in sd.items()}, f, metadata={"format": "pt"})
    res["adapter_keys"] = {"n": len(sd), "renamed": f"{VLLM_OLD}* -> {VLLM_NEW}*",
                           "example": VLLM_NEW + sorted(sd)[0][len(VLLM_OLD):]}
    for f in os.listdir(path):  # safetensors writes 0600 (gpu-lab qlora.py does the same);
        p = os.path.join(path, f)  # files only: the final save's listing holds checkpoint dirs
        if os.path.isfile(p):
            os.chmod(p, 0o644)


sampler.phase = "load"
res["index_keys_by_prefix"] = q38.index_prefixes()
model, info, load_s = q38.load()
res["load"] = {"seconds": round(load_s, 1), **q38.inventory(model, info),
               "torch_peak_allocated_GiB": gib(torch.cuda.max_memory_allocated())}
res["idle"] = [mem("after load")]
if info["missing_keys"] or res["load"]["param_devices"] != ["cuda:0"]:  # G4's rule, as Q1
    dump_run()
    sys.exit(f"STOP: missing keys {sorted(info['missing_keys'])[:10]} / devices {res['load']['param_devices']}")
sampler.phase = "peft"
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
sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
model.train()
gc.collect()
torch.cuda.empty_cache()
res["idle"].append(mem("LoRA attached"))
dump_run()
print(f"loaded in {load_s:.0f}s; {res['lora']}; {res['schedule']}; {res['idle'][-1]}", flush=True)

step, oom = 0, None
for epoch in range(EPOCHS):
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(SEED + epoch)).tolist()
    for s in range(steps_per_epoch):
        if sampler.abort or (SMOKE and step >= SMOKE):
            break
        batch = order[s * PER_STEP:(s + 1) * PER_STEP]
        n_step = sum(n_items[k] for k in batch)
        phase = f"step{step + 1}"
        non_torch = pynvml.nvmlDeviceGetMemoryInfo(nvh).used - torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        sampler.phase = phase
        t0 = time.time()
        loss_step = 0.0
        try:
            for k in batch:
                if n_items[k] == 0:
                    continue
                x = torch.tensor([data[k]["ids"]], device=0)
                y = torch.tensor([data[k]["labels"]], device=0)
                out = model(input_ids=x, labels=y, use_cache=False)
                loss = out.loss * (n_items[k] / n_step)  # token-mean over the whole step
                loss.backward()
                loss_step += loss.item()
                del out, loss
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        except torch.OutOfMemoryError as e:
            oom = {"step": step + 1, "epoch": epoch, "message": str(e)[:800],
                   "torch_peak_allocated_GiB": gib(torch.cuda.max_memory_allocated()),
                   "torch_peak_reserved_GiB": gib(torch.cuda.max_memory_reserved())}
            print(f"OOM at step {step + 1}: {str(e)[:300]}", flush=True)
            break
        step += 1
        torch.cuda.synchronize()
        dt = time.time() - t0
        p = sampler.peaks.get(phase, {})
        toks = sum(len(data[k]["ids"]) for k in batch)
        res["steps"].append({"step": step, "epoch": epoch, "loss": round(loss_step, 4),
                             "grad_norm": round(gnorm, 4), "lr": sched.get_last_lr()[0],
                             "seconds": round(dt, 2), "tokens": toks, "assistant_tokens": n_step,
                             "torch_peak_GiB": gib(torch.cuda.max_memory_allocated()),
                             "torch_peak_reserved_GiB": gib(torch.cuda.max_memory_reserved()),
                             "nvml_peak_GiB": gib(p.get("nvml_used", 0)),
                             "device_peak_GiB": gib(max(p.get("nvml_used", 0),
                                                        non_torch + torch.cuda.max_memory_reserved())),
                             "temp_C": p.get("temp_C"), "watts": round(p.get("watts", 0.0), 1),
                             "reasons": {r: v for r, v in p.get("reasons", {}).items() if v}})
        if step % 5 == 0 or step == 1:
            print(f"step {step}/{total_steps} loss {loss_step:.4f} gnorm {gnorm:.3f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {dt:.1f}s {toks / dt:.0f} tok/s "
                  f"peak {res['steps'][-1]['torch_peak_GiB']} dev {res['steps'][-1]['device_peak_GiB']} "
                  f"{p.get('temp_C')}C", flush=True)
        sampler.phase = "save"
        if step % CKPT_EVERY == 0 or (s == steps_per_epoch - 1 and not SMOKE):
            save(f"{OUT}/checkpoint-{step}")
        dump_run()
    if sampler.abort or oom or (SMOKE and step >= SMOKE):
        break

sampler.phase = "save"
res["oom"] = oom
if oom or sampler.abort:
    opt.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    save(f"{OUT}-partial")
    print(f"STOPPED ({'OOM' if oom else sampler.abort}) after step {step}; adapter so far in {OUT}-partial",
          flush=True)
else:
    save(OUT)
res["final_step"] = step
losses = [r["loss"] for r in res["steps"]]
if losses:
    res["loss_first"], res["loss_last"] = losses[0], losses[-1]
sampler.stop = True
dump_run()
print(f"done: {step} steps, loss {res.get('loss_first')} -> {res.get('loss_last')}, "
      f"abort={sampler.abort}, oom={bool(oom)}", flush=True)
sys.exit(3 if oom else 0)

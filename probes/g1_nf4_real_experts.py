"""G1: NF4 round trip on Lightning's real expert weights, and dequantize speed.

For each of the 23 MoE layers, stream the 128 up_proj and 128 down_proj
tensors from the shards into one fused (128, out, in) bf16 stack on the GPU
(the same layout the Route 1 loader builds) and quantize it with the exact
call replace_parameter_4bit makes: F.quantize_4bit(W, blocksize=None -> 64,
compress_statistics=True, quant_type="nf4"). Dequantize and compare.

Yardstick: that layer's shared_experts up/down weights quantized the way
Linear4bit does it under the G4 BitsAndBytesConfig (Params4bit: blocksize
64, compress_statistics = bnb_4bit_use_double_quant = True, nf4), i.e. the
same F.quantize_4bit call on each 2D weight. That is the error standard
QLoRA already accepts on every Linear.

Control (not gated): each expert quantized on its own, as per-expert
Linear4bit (Route 2) would. Double quantization subtracts the mean absmax
of the whole tensor, so fusing 128 experts into one tensor could in
principle change the error; this shows whether it does.

Error = relative Frobenius ||W - deq(q(W))|| / ||W||, accumulated in fp32
per expert; "overall" = sqrt(sum err^2 / sum W^2) over everything counted.

Dequantize speed: one full MoE layer (up + down stacks, 2.55 GB of bf16
output), F.dequantize_4bit(packed, state) as the parametrization calls it
(allocating its output), median of 20 runs after 3 warm-ups, CUDA events.

Pass (CHOSEN in docs/plan.md, fixed before this ran):
  packed bytes == numel / 2 for every stack;
  statistics bytes within 1% of the plan's arithmetic (numel/64 uint8
    absmax + 4 B per 256 blocks);
  overall expert error <= 1.5 x overall shared-expert error.

Run (laptop):
  docker run --rm --init --gpus all --ipc=host -v /srv/model-cache:/hf:ro \
    -v ~/moe-qlora/probes:/probes:ro -v ~/moe-qlora/results:/out \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 \
    --user $(id -u):$(id -g) --entrypoint python3 gpu-lab:training \
    /probes/g1_nf4_real_experts.py g1-laptop
"""
import json
import os
import statistics
import sys
import time

import bitsandbytes as bnb
import bitsandbytes.functional as F
import pynvml
import torch
from safetensors import safe_open

REV = "a9904d24bcc1d289a1950fa9d2b978c47cf903b9"
SNAP = os.path.join(os.environ.get("HF_HOME", "/hf"), "hub",
                    "models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "snapshots", REV)
label = sys.argv[1] if len(sys.argv) > 1 else "g1"
dev = torch.device("cuda", 0)
N_EXPERTS = 128
TIME_LAYER = 1  # first MoE layer; dequantize speed does not depend on the values

pynvml.nvmlInit()
nvh = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get("NVML_INDEX", "0")))
max_temp = 0

with open(os.path.join(SNAP, "model.safetensors.index.json")) as f:
    wmap = json.load(f)["weight_map"]
moe_layers = sorted({int(k.split(".")[2]) for k in wmap
                     if k.startswith("backbone.layers.") and ".mixer.experts." in k})
handles = {}


def get(name):
    shard = wmap[name]
    if shard not in handles:
        handles[shard] = safe_open(os.path.join(SNAP, shard), framework="pt", device="cpu")
    return handles[shard].get_tensor(name)


def qs_bytes(qs):  # same accounting as probes/g4_route1_load.py
    n = 0
    for t in (qs.absmax, qs.code, getattr(qs, "offset", None)):
        if isinstance(t, torch.Tensor):
            n += t.numel() * t.element_size()
    if getattr(qs, "state2", None) is not None:
        for t in (qs.state2.absmax, qs.state2.code):
            if isinstance(t, torch.Tensor):
                n += t.numel() * t.element_size()
    return n


def plan_stats_bytes(numel):  # docs/plan.md storage arithmetic
    blocks = numel // 64
    return blocks + 4 * ((blocks + 255) // 256)


def quantize(w):
    return F.quantize_4bit(w, blocksize=None, compress_statistics=True, quant_type="nf4")


def sq_errs(w, d):
    """Per leading-index fp32 sums of err^2 and W^2 (w, d: (E, out, in) or (out, in))."""
    if w.ndim == 2:
        w, d = w[None], d[None]
    e2 = torch.empty(w.shape[0], dtype=torch.float64)
    r2 = torch.empty(w.shape[0], dtype=torch.float64)
    for i in range(w.shape[0]):
        wf = w[i].float()
        e2[i] = (d[i].float() - wf).pow(2).sum().item()
        r2[i] = wf.pow(2).sum().item()
    return e2, r2


totals = {k: [0.0, 0.0] for k in ("expert", "shared", "expert_per_expert_control")}
per_layer = []
bytes_ = {"expert_numel": 0, "expert_packed": 0, "expert_stats": 0, "expert_stats_plan": 0,
          "packed_is_half_numel": True}
read_s = 0.0
timing = None
t_start = time.time()

for L in moe_layers:
    row = {"layer": L}
    for proj in ("up_proj", "down_proj"):
        shape0 = get(f"backbone.layers.{L}.mixer.experts.0.{proj}.weight").shape
        t0 = time.time()
        W = torch.empty((N_EXPERTS, *shape0), dtype=torch.bfloat16, device=dev)
        for e in range(N_EXPERTS):
            W[e].copy_(get(f"backbone.layers.{L}.mixer.experts.{e}.{proj}.weight"))
        torch.cuda.synchronize()
        read_s += time.time() - t0

        packed, qs = quantize(W)
        numel = W.numel()
        pb, sb = packed.numel() * packed.element_size(), qs_bytes(qs)
        bytes_["expert_numel"] += numel
        bytes_["expert_packed"] += pb
        bytes_["expert_stats"] += sb
        bytes_["expert_stats_plan"] += plan_stats_bytes(numel)
        bytes_["packed_is_half_numel"] &= (pb == numel // 2)

        D = F.dequantize_4bit(packed, qs)
        e2, r2 = sq_errs(W, D)
        del D
        rel_e = (e2 / r2).sqrt()
        totals["expert"][0] += e2.sum().item()
        totals["expert"][1] += r2.sum().item()

        # Control: every expert quantized on its own (per-expert Linear4bit, Route 2).
        c2 = torch.empty(N_EXPERTS, dtype=torch.float64)
        for e in range(N_EXPERTS):
            pe, qe = quantize(W[e])
            c2[e] = (F.dequantize_4bit(pe, qe).float() - W[e].float()).pow(2).sum().item()
        totals["expert_per_expert_control"][0] += c2.sum().item()
        totals["expert_per_expert_control"][1] += r2.sum().item()

        row[proj] = {
            "shape": list(W.shape),
            "rel_err_stack": round((e2.sum() / r2.sum()).sqrt().item(), 5),
            "rel_err_per_expert_min_median_max": [round(rel_e.min().item(), 5),
                                                  round(rel_e.median().item(), 5),
                                                  round(rel_e.max().item(), 5)],
            "rel_err_per_expert_control": round((c2.sum() / r2.sum()).sqrt().item(), 5),
            "w_absmax": W.abs().max().item(),
            "packed_bytes": pb, "stats_bytes": sb,
            "nested_offset": float(qs.offset) if getattr(qs, "offset", None) is not None else None,
        }

        if L == TIME_LAYER:
            if timing is None:
                timing = {"stacks": []}
            timing["stacks"].append((packed, qs, numel))
        else:
            del packed, qs
        del W

        # Yardstick: this layer's shared expert, as Linear4bit quantizes it.
        s = get(f"backbone.layers.{L}.mixer.shared_experts.{proj}.weight").to(dev)
        ps, qss = quantize(s)
        se2, sr2 = sq_errs(s, F.dequantize_4bit(ps, qss))
        totals["shared"][0] += se2.sum().item()
        totals["shared"][1] += sr2.sum().item()
        row[f"shared_{proj}"] = {"shape": list(s.shape),
                                 "rel_err": round((se2.sum() / sr2.sum()).sqrt().item(), 5)}
        del s, ps, qss

    if L == TIME_LAYER:  # one full layer = up + down stacks, dequantized as the forward does
        stacks = timing.pop("stacks")
        out_bytes = sum(n for _, _, n in stacks) * 2

        def one():
            for p, q, _ in stacks:
                F.dequantize_4bit(p, q)

        for _ in range(3):
            one()
        ms = []
        for _ in range(20):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            one()
            b.record()
            torch.cuda.synchronize()
            ms.append(a.elapsed_time(b))
        med = statistics.median(ms)
        timing = {"layer": L, "bf16_output_bytes": out_bytes, "median_ms": round(med, 3),
                  "min_ms": round(min(ms), 3), "max_ms": round(max(ms), 3),
                  "bf16_output_GB_per_s": round(out_bytes / 1e9 / (med / 1e3), 1)}
        del stacks
    torch.cuda.empty_cache()
    max_temp = max(max_temp, pynvml.nvmlDeviceGetTemperature(nvh, pynvml.NVML_TEMPERATURE_GPU))
    per_layer.append(row)
    print(f"layer {L}: up {row['up_proj']['rel_err_stack']} down {row['down_proj']['rel_err_stack']} "
          f"shared {row['shared_up_proj']['rel_err']}/{row['shared_down_proj']['rel_err']} "
          f"t={time.time() - t_start:.0f}s", flush=True)


def overall(k):
    e2, r2 = totals[k]
    return round((e2 / r2) ** 0.5, 5)


err = {k: overall(k) for k in totals}
ratio = round(err["expert"] / err["shared"], 4)
stats_dev = bytes_["expert_stats"] / bytes_["expert_stats_plan"] - 1
checks = {
    "packed_bytes_eq_half_numel": bytes_["packed_is_half_numel"],
    "stats_within_1pct_of_plan": abs(stats_dev) <= 0.01,
    "expert_err_le_1p5x_shared": ratio <= 1.5,
}
res = {
    "label": label, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "bitsandbytes": bnb.__version__,
    "snapshot": SNAP, "n_moe_layers": len(moe_layers),
    "quantize_call": "F.quantize_4bit(W, blocksize=None->64, compress_statistics=True, quant_type='nf4')",
    "bytes": {**bytes_, "stats_deviation_from_plan": round(stats_dev, 6),
              "expert_packed_GB": round(bytes_["expert_packed"] / 1e9, 4),
              "expert_stats_GB": round(bytes_["expert_stats"] / 1e9, 4)},
    "overall_rel_err": err, "expert_over_shared_ratio": ratio,
    "dequant_timing": timing,
    "read_seconds": round(read_s, 1), "total_seconds": round(time.time() - t_start, 1),
    "max_temp_C": max_temp,
    "checks": checks, "pass": all(checks.values()),
    "per_layer": per_layer,
}
print(json.dumps({k: v for k, v in res.items() if k != "per_layer"}, indent=1))
with open(f"/out/{label}.json", "w") as f:
    json.dump(res, f, indent=1)

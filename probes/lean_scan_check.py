"""Check lean_scan.mamba2_chunk_scan against the 5.16.1 torch original, on the GPU.

Pass (CHOSEN 2026-09-24, before the first run): for the output, the final state and the
gradient of every input, max |lean - original| <= 1e-4 x max |original|. fp32 sums
reordered into matmuls should land near 1e-6; 1e-4 still catches a wrong index.

Inputs use Lightning's shapes (64 heads, head dim 64, 8 groups, state 128, chunk 128),
bf16 where the mixer passes bf16, with random values. Cases: seq 300 (padding path) with
initial states, and seq 1024. Then the scan's own peak memory, forward + backward, for
both versions at seq 1024 and 2048 (an OOM is recorded, not retried).

Run: docker run --rm --gpus all -v ~/moe-qlora/probes:/probes:ro -v ~/moe-qlora/results:/out \
       --user $(id -u):$(id -g) --entrypoint python3 gpu-lab:training /probes/lean_scan_check.py LABEL [fp32]

Added after the first run (lean-scan-check-desktop.json) failed the line above: every
fp32 quantity agreed to ~1e-7, and every failure was the gradient of a bf16 input, off
by exactly one bf16 step for its magnitude. Two additions test that reading without
moving the line:
  - `fp32` as the second argument makes every input fp32, so no gradient is rounded
    to bf16. Same 1e-4 line. This tests the math on its own.
  - For bf16 tensors, `max_ulp` counts the largest difference in bf16 steps (exact, from
    the bit patterns). It describes the result and is not a pass line.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_scan  # noqa: E402

label = sys.argv[1]
FP32 = len(sys.argv) > 2 and sys.argv[2] == "fp32"
H, P, G, N, CHUNK = 64, 64, 8, 128, 128
TOL = 1e-4
dev = "cuda"


def inputs(T, seed, init_states):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    t = {"hidden_states": r(1, T, H, P).bfloat16(), "dt": (r(1, T, H) - 3).bfloat16(),
         "A": -torch.exp(torch.rand(H, generator=g) * 2.77), "B": r(1, T, G, N).bfloat16(),
         "C": r(1, T, G, N).bfloat16(), "D": r(H).bfloat16(), "dt_bias": (r(H) * 0.5).bfloat16()}
    if init_states:
        t["initial_states"] = r(1, H, P, N)
    if FP32:
        t = {k: v.float() for k, v in t.items()}
    return {k: v.to(dev).requires_grad_(True) for k, v in t.items()}


def max_ulp(ref, new):
    """Largest difference in bf16 steps; same-sign bf16 bit patterns are ordered."""
    a, b = ref.view(torch.int16).int(), new.view(torch.int16).int()
    same_sign = (a < 0) == (b < 0)
    return int((a - b).abs()[same_sign].max().item()), int((~same_sign & (ref != new)).sum().item())


def run(fn, t, seed):
    kw = {k: v for k, v in t.items() if k not in ("hidden_states", "dt", "A", "B", "C")}
    out, final = fn(t["hidden_states"], t["dt"], t["A"], t["B"], t["C"], chunk_size=CHUNK,
                    dt_softplus=True, return_final_states=True, z=None, **kw)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    loss = (out.float() * torch.randn(out.shape, generator=g).to(dev)).sum() \
        + (final.float() * torch.randn(final.shape, generator=g).to(dev)).sum()
    grads = torch.autograd.grad(loss, list(t.values()))
    return {"output": out.detach(), "final_state": final.detach(),
            **{f"grad_{k}": gr.detach() for k, gr in zip(t, grads)}}


res = {"label": label, "inputs": "fp32" if FP32 else "bf16 as the mixer passes",
       "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
       "modeling_sha256": lean_scan.modeling_sha256(), "tolerance": TOL,
       "matmul_precision": torch.get_float32_matmul_precision(),
       "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "cases": {}}
assert res["modeling_sha256"] == lean_scan.EXPECTED_SHA256

for T, seed, init in [(300, 1, True), (1024, 2, False)]:
    t = inputs(T, seed, init)
    ref = run(lean_scan.ORIGINAL, t, seed)
    new = run(lean_scan.mamba2_chunk_scan, t, seed)
    case = {}
    for k in ref:
        scale = ref[k].float().abs().max().item()
        diff = (new[k].float() - ref[k].float()).abs().max().item()
        case[k] = {"max_abs_ref": scale, "max_abs_diff": diff, "rel": diff / scale if scale else diff,
                   "same_shape_dtype": ref[k].shape == new[k].shape and ref[k].dtype == new[k].dtype,
                   "dtype": str(ref[k].dtype)}
        if ref[k].dtype == torch.bfloat16:
            case[k]["max_ulp"], case[k]["sign_flips"] = max_ulp(ref[k], new[k])
            case[k]["n_elements_differing"] = int((ref[k] != new[k]).sum().item())
            case[k]["n_elements"] = ref[k].numel()
    res["cases"][f"seq{T}" + ("_init_states" if init else "")] = case
    del t, ref, new
    torch.cuda.empty_cache()

res["pass"] = all(v["rel"] <= TOL and v["same_shape_dtype"] for c in res["cases"].values() for v in c.values())

res["peak_scan_fwd_bwd_GiB"] = {}
for T in (1024, 2048):
    for name, fn in (("original", lean_scan.ORIGINAL), ("lean", lean_scan.mamba2_chunk_scan)):
        t = inputs(T, 3, False)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            run(fn, t, 3)
            torch.cuda.synchronize()
            v = {"GiB": round((torch.cuda.max_memory_allocated() - base) / 2**30, 3),
                 "seconds": round(time.perf_counter() - t0, 3)}
        except torch.OutOfMemoryError as e:
            v = {"oom": str(e)[:200]}
        res["peak_scan_fwd_bwd_GiB"][f"seq{T}_{name}"] = v
        del t
        torch.cuda.empty_cache()

with open(f"/out/{label}.json", "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps({"pass": res["pass"], "worst_rel": max(v["rel"] for c in res["cases"].values() for v in c.values()),
                  "peak": res["peak_scan_fwd_bwd_GiB"], "tf32": res["tf32_matmul"]}, indent=1))

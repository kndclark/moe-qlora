"""Row P probe: can the CPU compute a cold expert faster than the copy engine can bring it to the GPU
(~110 us an expert on the laptop)? cpu_expert.c reads NVFP4 straight from the checkpoint's
layout; this checks it against a float64 dequant in torch, then (--time, idle machine only) times
decode-shaped calls, experts from a pool bigger than L3 so every call reads RAM, and the RAM read
rate itself. CPU only; in the vLLM image with the HF cache at /hf:
  docker run --rm --pull never --entrypoint python3 -v /srv/model-cache:/hf:ro -v $PWD:/p \\
    vllm/vllm-openai:v0.29.0 /p/cpu_expert_probe.py [--threads 2] [--time --threads 1,8,16,24]"""
import argparse
import ctypes
import json
import os
import random
import subprocess
import time

import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--snap", default="/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/"
                "bee7596271d1495f6992ae224aefde4410e816b8")
ap.add_argument("--layer", type=int, default=1)
ap.add_argument("--experts", type=int, default=8)
ap.add_argument("--threads", default="2")
ap.add_argument("--time", action="store_true", help="timing and RAM read rate: only with nothing else running")
ap.add_argument("--pool-gib", type=float, default=2.0, help="--time: expert copies to cycle through (> L3)")
a = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
so = "/tmp/cpu_expert.so"
subprocess.run(["gcc", "-O3", "-mavx2", "-mfma", "-shared", "-fPIC", f"{here}/cpu_expert.c", "-o", so, "-lpthread",
                "-lm"], check=True)
lib = ctypes.CDLL(so)
lib.ex_pool.restype = ctypes.c_void_p
lib.ex_pool.argtypes = [ctypes.c_int]
lib.ex_pool_free.argtypes = [ctypes.c_void_p]
lib.ex_moe.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 7
lib.ex_read.restype = ctypes.c_double
lib.ex_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]


class Ex(ctypes.Structure):
    _fields_ = [("uw", ctypes.c_void_p), ("us", ctypes.c_void_p), ("dw", ctypes.c_void_p), ("ds", ctypes.c_void_p),
                ("ug", ctypes.c_float), ("dg", ctypes.c_float)]


wm = json.load(open(f"{a.snap}/model.safetensors.index.json"))["weight_map"]
files = {}


def get(name):
    f = wm[name]
    if f not in files:
        files[f] = safe_open(f"{a.snap}/{f}", framework="pt")
    return files[f].get_tensor(name)


experts = []
for e in range(a.experts):
    p = f"backbone.layers.{a.layer}.mixer.experts.{e}."
    d = {}
    for k in ("up", "down"):
        d[k + "w"] = get(p + f"{k}_proj.weight").contiguous()
        d[k + "s"] = get(p + f"{k}_proj.weight_scale").contiguous()
        d[k + "g"] = float(get(p + f"{k}_proj.weight_scale_2").float().reshape(-1)[0])
    experts.append(d)
I, H2 = experts[0]["upw"].shape
H = 2 * H2
x0 = experts[0]
print(f"layer {a.layer}: {len(experts)} experts, up {tuple(x0['upw'].shape)} {x0['upw'].dtype} scale "
      f"{tuple(x0['ups'].shape)} {x0['ups'].dtype}, down {tuple(x0['downw'].shape)} scale {tuple(x0['downs'].shape)}; "
      f"global scales up {x0['upg']:.3g} down {x0['downg']:.3g}", flush=True)
ebytes = sum(x0[k].numel() * x0[k].element_size() for k in ("upw", "ups", "downw", "downs"))

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)


def deq(w, s, g):
    """NVFP4 as vLLM's emulation reads it: low nibble first, e4m3 scale per 16, times the global scale."""
    q = torch.stack((w & 15, w >> 4), dim=-1).reshape(w.shape[0], -1).long()
    return E2M1[q] * s.to(torch.float64).repeat_interleave(16, dim=1) * g


try:  # the same nibble order and table as vLLM's own emulation?
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import break_fp4_bytes
    w = x0["upw"][:64]
    q = torch.stack((w & 15, w >> 4), dim=-1).reshape(64, -1).long()
    print("vLLM break_fp4_bytes matches the table:", torch.equal(break_fp4_bytes(w, torch.float32).double(), E2M1[q]))
except Exception as ex:  # noqa: BLE001
    print("vLLM emulation check skipped:", type(ex).__name__, ex)

ref_w = [{k: deq(d[k + "w"], d[k + "s"], d[k + "g"]) for k in ("up", "down")} for d in experts]


def structs(ds):
    arr = (Ex * len(ds))()
    for i, d in enumerate(ds):
        arr[i] = Ex(d["upw"].data_ptr(), d["ups"].data_ptr(), d["downw"].data_ptr(), d["downs"].data_ptr(),
                    d["upg"], d["downg"])
    return arr


def call(pool, ds, counts, T, x, y, seed=0):
    """Expert i serves counts[i] tokens (random, distinct within an expert) with random weights."""
    rng = random.Random(seed)
    tok, off = [], [0]
    for c in counts:
        tok += rng.sample(range(T), c)
        off.append(len(tok))
    tw = torch.tensor([rng.uniform(0.05, 1.0) for _ in tok], dtype=torch.float32)
    tok_t, off_t = torch.tensor(tok, dtype=torch.int32), torch.tensor(off, dtype=torch.int32)
    h = torch.empty(len(tok), I, dtype=torch.float32)
    arr = structs(ds)
    args = (pool, H, I, len(ds), arr, off_t.data_ptr(), tok_t.data_ptr(), tw.data_ptr(), x.data_ptr(), y.data_ptr(),
            h.data_ptr())
    return args, (tok, off, tw), (arr, tok_t, off_t, tw, h, x, y)  # keep every buffer the call writes or reads


def reference(idx, plan, x):
    tok, off, tw = plan
    y = torch.zeros(x.shape, dtype=torch.float64)
    xd = x.double()
    for i, e in enumerate(idx):
        for j in range(off[i], off[i + 1]):
            hv = torch.relu(ref_w[e]["up"] @ xd[tok[j]]) ** 2
            y[tok[j]] += float(tw[j]) * (ref_w[e]["down"] @ hv)
    return y


threads = [int(t) for t in a.threads.split(",")]
torch.manual_seed(0)
T = 24
x = torch.randn(T, H, dtype=torch.float32)
for nt in threads:
    pool = lib.ex_pool(nt)
    for counts in ([1], [2, 3], [4, 5, 9, 1, 6, 7, 8, 1]):
        idx = list(range(len(counts)))
        y0 = torch.randn(T, H, dtype=torch.float32)
        y = y0.clone()
        args, plan, keep = call(pool, [experts[e] for e in idx], counts, T, x, y, seed=len(counts))
        assert lib.ex_moe(*args) == 0
        ref = reference(idx, plan, x)
        err = ((y.double() - y0.double()) - ref).abs().max().item() / ref.abs().max().item()
        print(f"threads {nt}, tokens per expert {counts}: max error {err:.2e} of max |y| {ref.abs().max().item():.3g}"
              f" ({'ok' if err < 1e-4 else 'BAD'})", flush=True)
    lib.ex_pool_free(pool)

if not a.time:
    raise SystemExit
# Timing: a pool of expert copies bigger than L3 (36 MiB on the 275HX), cycled so every call reads RAM.
ncopy = max(1, int(a.pool_gib * 2 ** 30 / ebytes / len(experts)))
big = [{k: (d[k].clone() if isinstance(d[k], torch.Tensor) else d[k]) for k in d} for _ in range(ncopy) for d in experts]
print(f"pool: {len(big)} experts, {len(big) * ebytes / 2 ** 30:.2f} GiB; an expert is {ebytes / 2 ** 20:.2f} MiB",
      flush=True)
buf = torch.empty(int(a.pool_gib * 2 ** 30) // 128 * 128, dtype=torch.uint8)
buf.fill_(1)
for nt in threads:
    pool = lib.ex_pool(nt)
    lib.ex_read(pool, buf.data_ptr(), buf.numel())
    t0 = time.perf_counter()
    for _ in range(3):
        lib.ex_read(pool, buf.data_ptr(), buf.numel())
    rd = 3 * buf.numel() / (time.perf_counter() - t0) / 1e9
    line = [f"threads {nt}: RAM read {rd:.1f} GB/s"]
    for ne, per in ((1, 1), (2, 1), (6, 1), (12, 1), (24, 1), (12, 4), (6, 16)):
        cur, n = 0, 40
        calls = []
        for i in range(n):
            ds = [big[(cur + j) % len(big)] for j in range(ne)]
            cur += ne
            calls.append(call(pool, ds, [per] * ne, 64, torch.randn(64, H), torch.zeros(64, H), seed=i))
        for c in calls[:3]:
            lib.ex_moe(*c[0])
        t0 = time.perf_counter()
        for c in calls:
            lib.ex_moe(*c[0])
        us = (time.perf_counter() - t0) / n * 1e6
        line.append(f"{ne}x{per}: {us:.0f} us ({us / ne:.0f} us/expert, {ne * ebytes / us / 1e3:.1f} GB/s)")
    print("; ".join(line), flush=True)
    lib.ex_pool_free(pool)

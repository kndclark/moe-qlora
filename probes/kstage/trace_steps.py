"""Row L: where an agent-load step's wall goes, from trace_agents.py's torch-profiler windows.
Kernels inside a CUDA graph run on several streams at once (the shared expert and the LoRA beside
the routed experts), so summing kernel time per family overstates wall. Instead each step (cut at
the sampler kernel) is cut at its residual-add RMSNorm kernels, which every layer starts with and
which wait for the layer before: 53 per step, 52 layers plus the final norm. A segment's wall,
idle gaps included, goes to the layer type of the kernel that marks it (Marlin MoE = moe, the SSM
scan or conv = mamba, FlashInfer or kernel_mha = attention); what precedes the first norm and
follows the last one (input prep, lm_head, sampler, host KV and K6 copies) is "step". Steps group
by the runner's annotation: context tokens (prefill, chunked) and generation sequences (decode).
usage: trace_steps.py TRACE_DIR_OR_GZ... [--top 0]
(results/kv-levers/trace-p39-*; windows pair with the "window" lines of bench-p39-*.log)"""
import argparse
import bisect
import glob
import gzip
import json
import os
import re
from collections import defaultdict

ANCHORS = [  # first match wins; read off the p39 traces
    ("moe", re.compile(r"marlin_moe|fused_moe")),
    ("attention", re.compile(r"kernel_mha|flashinfer|BatchPrefill|BatchDecode|fmha|flash_fwd")),
    ("mamba", re.compile(r"selective_scan|chunk_scan|chunk_state|state_passing|causal_conv1d|ssd_")),
]
KINDS = ["mamba", "moe", "attention", "step"]
ANN = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


def kind(seg):
    for n, p in ANCHORS:
        if any(p.search(e["name"]) for e in seg):
            return n
    return "step"


def window(path, top):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        ev = json.load(f)["traceEvents"]
    ks = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
                key=lambda e: e["ts"])
    if not ks:
        print(f"{os.path.basename(path)}: no kernels")
        return
    ann = sorted((e for e in ev if e.get("cat") == "gpu_user_annotation" and ANN.match(e["name"])),
                 key=lambda e: e["ts"])
    ats = [e["ts"] for e in ann]
    cut = [i for i, e in enumerate(ks) if "_gumbel_sample" in e["name"]]
    span = ks[-1]["ts"] + ks[-1]["dur"] - ks[0]["ts"]
    cls = defaultdict(lambda: {"n": 0, "wall": [], "tok": 0, **{k: 0.0 for k in KINDS}})
    copies = defaultdict(float)
    for a, b in zip(cut, cut[1:]):  # a step: after one sampler through the next
        st = ks[a + 1:b + 1]
        t0, t1 = ks[a]["ts"] + ks[a]["dur"], st[-1]["ts"] + st[-1]["dur"]
        j = bisect.bisect_right(ats, t1) - 1  # the annotation piece covering the step's end
        while j >= 0 and ann[j]["ts"] + ann[j]["dur"] < t0:
            j = -1
        m = ANN.match(ann[j]["name"]) if j >= 0 else None
        ctx, gen = (int(m.group(2)), int(m.group(4))) if m else (-1, -1)
        key = ("prefill" if ctx > 0 else "decode", ctx, gen)
        c = cls[key]
        c["n"] += 1
        c["wall"].append(t1 - t0)
        c["tok"] += max(ctx, 0) + max(gen, 0)
        nb = [i for i, e in enumerate(st) if "rms_norm" in e["name"]]
        edges = [t0] + [st[i]["ts"] for i in nb] + [t1]
        segs = [st[:nb[0]] if nb else st] + [st[i:k] for i, k in zip(nb, nb[1:] + [len(st)])]
        for i, seg in enumerate(segs):
            k = "step" if i in (0, len(segs) - 1) else kind(seg)  # the final norm's segment holds lm_head
            c[k] += edges[i + 1] - edges[i]
        for e in st:
            if e["cat"] != "kernel":
                copies[e["name"]] += e["dur"]
    stepped = sum(sum(c["wall"]) for c in cls.values())
    print(f"{os.path.basename(path)[:40]}: span {span / 1e3:.0f} ms, {len(cut) - 1} whole steps "
          f"cover {stepped / 1e3:.0f} ms")
    for key, c in sorted(cls.items(), key=lambda x: -sum(x[1]["wall"])):
        w = sorted(c["wall"])
        tot = sum(w)
        split = ", ".join(f"{k} {100 * c[k] / tot:.0f}%" for k in KINDS)
        print(f"  {key[0]:7s} ctx {key[1]:5d} gen {key[2]:2d}: {c['n']:4d} steps, median {w[len(w) // 2] / 1e3:7.1f} ms, "
              f"{1e6 * c['tok'] / tot:6.0f} tok/s | {split}")
    if copies:
        print("  copies: " + "; ".join(f"{n} {t / 1e3:.1f} ms" for n, t in sorted(copies.items(), key=lambda x: -x[1])))
    if top:
        names = defaultdict(float)
        for e in ks:
            names[e["name"][:100]] += e["dur"]
        for n, t in sorted(names.items(), key=lambda x: -x[1])[:top]:
            print(f"    {t / 1e3:7.1f} ms  {n}")


ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+")
ap.add_argument("--top", type=int, default=0, help="also list the top kernels by GPU time")
a = ap.parse_args()
for p in a.paths:
    for f in sorted(glob.glob(os.path.join(p, "*.gz")), key=os.path.getmtime) if os.path.isdir(p) else [p]:
        window(f, a.top)

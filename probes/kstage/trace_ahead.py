"""Row O, lever B1: where KSTAGE_AHEAD's ~31 us a layer goes, from trace_drive.py's torch
profiler traces of steady decode (kernels inside CUDA graphs carry GPU start and end times).
A decode step runs from one `_gumbel_sample_kernel` (the sampler) to the next. Per step: the
period, the time at least one kernel runs (union over streams: a graph's branches land on two or
three stream ids), and the GPU time of every kernel name. The arms are then diffed name by name,
so the step's growth splits into added kernel time and added idle time (stream waits, launch
gaps, and the captured cuStreamWaitValue32 / WriteValue32, which are not kernels). Copies on
streams that run no kernels (ce_helper's own context) are reported apart.
usage: trace_ahead.py BASE_DIR AHEAD_DIR [--layers 23] [--top 14]
(results/kv-levers/trace-p19-trace-{base,ahead}; traces pair up in time order, one per
concurrency trace_drive.py ran)"""
import argparse
import glob
import gzip
import json
import os
import statistics
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("ahead")
ap.add_argument("--layers", type=int, default=23, help="MoE layers a step")
ap.add_argument("--top", type=int, default=14)
a = ap.parse_args()


def steps(path):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        ev = json.load(f)["traceEvents"]
    ks = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"), key=lambda e: e["ts"])
    ktid = {e["tid"] for e in ks}
    side = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "gpu_memcpy" and e["tid"] not in ktid]
    marks = [e["ts"] for e in ks if e["name"].startswith("_gumbel_sample")]
    out, i = [], 0
    for t0, t1 in zip(marks, marks[1:]):
        while i < len(ks) and ks[i]["ts"] < t0:
            i += 1
        j, names, busy, end = i, defaultdict(float), 0.0, t0
        while j < len(ks) and ks[j]["ts"] < t1:
            k = ks[j]
            names[k["name"][:72]] += k["dur"]
            s, e = k["ts"], min(k["ts"] + k["dur"], t1)
            if e > end:
                busy += e - max(s, end)
                end = e
            j += 1
        out.append((t1 - t0, busy, names))
    sc = [e for e in side if marks and marks[0] <= e["ts"] < marks[-1]]
    return out, sc


def summary(st):
    per = statistics.median(s[0] for s in st)
    keep = [s for s in st if s[0] < 1.5 * per]  # drop steps the scheduler stalled between
    n = len(keep)
    names = defaultdict(float)
    for _, _, nm in keep:
        for k, v in nm.items():
            names[k] += v / n
    return {"n": n, "dropped": len(st) - n, "period": sum(s[0] for s in keep) / n, "median": per,
            "busy": sum(s[1] for s in keep) / n, "names": names}


def files(d):
    return sorted(glob.glob(os.path.join(d, "*.json*")), key=lambda f: int(os.path.basename(f).split(".")[1]))


L = a.layers
for fb, fa in zip(files(a.base), files(a.ahead)):
    (sb, _), (sa, side) = steps(fb), steps(fa)
    b, h = summary(sb), summary(sa)
    dp, db = h["period"] - b["period"], h["busy"] - b["busy"]
    print(f"== {os.path.basename(fb)[-30:]} vs {os.path.basename(fa)[-30:]}: steps {b['n']} (+{b['dropped']} dropped) "
          f"vs {h['n']} (+{h['dropped']})")
    print(f"  step period {b['period']:.0f} -> {h['period']:.0f} us (median {b['median']:.0f} -> {h['median']:.0f}): "
          f"{dp:+.0f} a step, {dp / L:+.1f} a layer")
    print(f"  GPU busy    {b['busy']:.0f} -> {h['busy']:.0f} us: {db:+.0f} a step, {db / L:+.1f} a layer; "
          f"idle {b['period'] - b['busy']:.0f} -> {h['period'] - h['busy']:.0f} us: {(dp - db) / L:+.1f} a layer")
    nb, nh = b["names"], h["names"]
    d = sorted(set(nb) | set(nh), key=lambda k: -abs(nh.get(k, 0) - nb.get(k, 0)))
    for k in d[:a.top]:
        x, y = nb.get(k, 0), nh.get(k, 0)
        print(f"    {(y - x) / L:+7.2f} us a layer  ({x / L:6.2f} -> {y / L:6.2f})  {k}")
    rest = sum(nh.get(k, 0) - nb.get(k, 0) for k in d[a.top:])
    print(f"    {rest / L:+7.2f} us a layer  every other kernel name")
    if side:
        n = h["n"] + h["dropped"]
        by = defaultdict(lambda: [0, 0.0, 0])
        for e in side:
            r = by[e["name"][:32]]
            r[0] += 1
            r[1] += e["dur"]
            r[2] += e.get("args", {}).get("bytes", 0)
        for k, (c, t, by_) in by.items():
            print(f"  side copies (ce_helper): {k}: {c / n:.2f} a step, {t / n:.0f} us, {by_ / n / 2**20:.2f} MiB a step")

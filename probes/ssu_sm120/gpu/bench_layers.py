"""Per-decode-step SSU time with Lightning's 23 Mamba layers, each with its
own state cache (so the 2 MiB-per-sequence fp32 state is not L2-resident
from the previous call, as in the real model). One step = 23 SSU calls.
Reports eager and CUDA-graph medians over >= 60 steps."""
import json
import statistics
import sys

import torch

sys.path.insert(0, "/work")
import common  # noqa: E402

N_MAMBA = 23
BATCHES = [int(b) for b in (sys.argv[1] if len(sys.argv) > 1 else "1,16,64").split(",")]
fns = {"triton": common.call_triton}
for a in ["simple", "vertical", "horizontal", "auto"]:
    fns[a] = (lambda inp, a=a: common.call_flashinfer(inp, a))


def step(fn, layers):
    for inp in layers:
        fn(inp)


def t_eager(fn, layers, n=60):
    for _ in range(5):
        step(fn, layers)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        step(fn, layers)
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000)
    return statistics.median(ts)


def t_graph(fn, layers, n=60):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step(fn, layers)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step(fn, layers)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000)
    del g
    return statistics.median(ts)


for b in BATCHES:
    layers = [common.make_inputs(b, nslots=b + 3, layout="paged", seed=100 + i)
              for i in range(N_MAMBA)]
    res = {}
    for name, fn in fns.items():
        e = t_eager(fn, layers)
        g = t_graph(fn, layers)
        res[name] = dict(step_eager_us=e, step_graph_us=g,
                         per_call_graph_us=g / N_MAMBA,
                         per_token_graph_us=g / b, per_token_eager_us=e / b)
    print("STEP " + json.dumps({"batch": b, "res": res}), flush=True)
    del layers
    torch.cuda.empty_cache()

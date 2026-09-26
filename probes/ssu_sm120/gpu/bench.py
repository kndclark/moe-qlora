"""Correctness (FlashInfer variants vs vLLM Triton SSU, plus an fp64 torch
reference) and speed (eager per-call events; CUDA-graph device time) at
Lightning's decode shapes. Also lists the kernel names each call launches."""
import json
import statistics
import sys

import torch
import torch.nn.functional as Fn

sys.path.insert(0, "/work")
import common  # noqa: E402

BATCHES = [int(b) for b in (sys.argv[1] if len(sys.argv) > 1 else "1,16,64").split(",")]
ALGOS = ["simple", "vertical", "horizontal", "auto"]
LAYOUTS = ["paged", "contig"]


def snapshot(inp):
    return inp["state"].clone(), inp["out"].clone()


def restore(inp, st, out):
    inp["state"].copy_(st)
    inp["out"].copy_(out)


def ref64(inp, st0):
    """fp64 reference of Mamba2 single-token SSU (dt_softplus=True, no z)."""
    idx = inp["idx"][:, 0].long()
    s = st0[idx].double()                                   # (b, h, d, n)
    x = inp["x"].double()                                   # (b, h, d)
    dt = Fn.softplus(inp["dt"].double() + inp["dt_bias"].double())  # (b, h, d)
    A = inp["A"].double()                                   # (h, d, n)
    rep = common.NH // common.NG
    B = inp["B"].double().repeat_interleave(rep, dim=1)     # (b, h, n)
    C = inp["C"].double().repeat_interleave(rep, dim=1)
    dA = torch.exp(A[None] * dt[..., None])
    s_new = s * dA + (dt * x)[..., None] * B[:, :, None, :]
    y = (s_new * C[:, :, None, :]).sum(-1) + inp["D"].double() * x
    return y, s_new, idx


def diffs(a, b):
    a = a.double()
    b = b.double()
    d = (a - b).abs()
    return dict(max_abs=float(d.max()),
                max_rel=float((d / b.abs().clamp_min(1e-6)).max()),
                max_abs_over_max_ref=float(d.max() / b.abs().max()),
                mean_abs=float(d.mean()))


def run(fn, inp):
    fn(inp)
    torch.cuda.synchronize()


def time_eager(fn, inp, iters=200, warm=30):
    for _ in range(warm):
        fn(inp)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        fn(inp)
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    return statistics.median(ts)


def time_graph(fn, inp, calls=20, replays=60):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn(inp)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn(inp)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(replays):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0 / calls)
    return statistics.median(ts)


def kernel_names(fn, inp):
    from torch.profiler import ProfilerActivity, profile
    try:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(inp)
            torch.cuda.synchronize()
        names = sorted({e.name for e in prof.events()
                        if e.device_type.name == "CUDA" and "emcpy" not in e.name
                        and "emset" not in e.name})
        return [n[:110] for n in names]
    except Exception as e:  # noqa: BLE001
        return [f"profiler failed: {type(e).__name__}: {e}"[:200]]


triton_fn = common.call_triton
fi_fns = {a: (lambda inp, a=a: common.call_flashinfer(inp, a)) for a in ALGOS}

for layout in LAYOUTS:
    for b in BATCHES:
        inp = common.make_inputs(b, nslots=b + 3, layout=layout, seed=b)
        st0, out0 = snapshot(inp)
        y64, s64, idx = ref64(inp, st0)
        # Triton reference
        run(triton_fn, inp)
        tri_out = inp["out"].clone()
        tri_state = inp["state"].clone()
        rows = {"triton": dict(
            vs_fp64_out=diffs(tri_out, y64), vs_fp64_state=diffs(tri_state[idx], s64),
            untouched_slots_changed=bool((tri_state != st0).any(dim=(1, 2, 3))[
                [i for i in range(st0.shape[0]) if i not in set(idx.tolist())]].any()))}
        for a, fn in fi_fns.items():
            restore(inp, st0, out0)
            run(fn, inp)
            o, s = inp["out"].clone(), inp["state"].clone()
            rows[a] = dict(
                nan_in_out=bool(torch.isnan(o.float()).any()),
                vs_triton_out=diffs(o, tri_out),
                vs_triton_state=diffs(s, tri_state),
                vs_fp64_out=diffs(o, y64), vs_fp64_state=diffs(s[idx], s64))
        print("CORR " + json.dumps({"layout": layout, "batch": b, "rows": rows}), flush=True)
        if layout != "paged":
            continue
        # which kernels actually run
        kn = {"triton": kernel_names(triton_fn, inp)}
        for a, fn in fi_fns.items():
            kn[a] = kernel_names(fn, inp)
        print("KERN " + json.dumps({"batch": b, "kernels": kn}), flush=True)
        # speed
        t = {}
        for name, fn in [("triton", triton_fn)] + list(fi_fns.items()):
            restore(inp, st0, out0)
            t[name] = dict(eager_us=time_eager(fn, inp), graph_us=time_graph(fn, inp))
        print("TIME " + json.dumps({"batch": b, "times": t}), flush=True)

"""K6's miss copy, alone: what each copy kernel costs per launch with P possible copies of
which K are live, rows in VMM host pages copied to device rows as in decode. p19's trace put
_copy_rows_kernel's floor at 12 / 43 / 168 us for P = 6 / 24 / 96 with nothing to copy.
Each case is a CUDA graph of --n launches (decode runs in graphs), each launch with its own
plan so no source row repeats within a replay (the L2 cannot serve the reads). Every live
row is checked after the replays.
usage (vLLM image): python3 /k/copy_rows_probe.py [--live 82,164,246,328] [--few 82]"""
import argparse
import sys

import torch
import triton

sys.path.insert(0, "/k")
import vllm_kstage as ks  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--P", default="6,24,96")
ap.add_argument("--K", default="0,1,3")
ap.add_argument("--few", default="82")
ap.add_argument("--live", default="82,164,246,328")
ap.add_argument("--n", type=int, default=20, help="launches a graph")
ap.add_argument("--reps", type=int, default=20, help="graph replays timed")
a = ap.parse_args()
ints = lambda s: [int(x) for x in s.split(",") if x]

ROW = 623_616 * 4  # one w13 row of Lightning's experts (168 x 3712 int32), bytes
E, ND, BS = 128, 64, 128  # rows 0..63 device pages, 64..127 host pages
torch.zeros(1, device="cuda")  # MixedRows needs a current context
mr = ks.MixedRows(E, ROW, ND)
t = mr.tensor(torch.int32)
W = t.shape[1]
g = torch.Generator(device="cuda").manual_seed(0)
for r in range(ND, E):  # random host rows; device rows hold garbage until copied
    t[r].copy_(torch.randint(-2**31, 2**31 - 1, (W,), device="cuda", dtype=torch.int32, generator=g))
torch.cuda.synchronize()
print(f"{torch.cuda.get_device_name()}: rows of {ROW / 2**20:.2f} MiB ({W} words), host rows {E - ND}, "
      f"BLOCK {ks.BLOCK}, warps {ks.WARPS}; us a launch, mean of {a.reps} replays of {a.n}", flush=True)


def plans(K):
    """--n plans, each K distinct host rows -> K distinct device rows, -1 after them."""
    out = []
    for i in range(a.n):
        cps = torch.full((BS,), -1, dtype=torch.int32, device="cuda")
        cpd = torch.zeros(BS, dtype=torch.int32, device="cuda")
        for k in range(K):
            cps[k] = ND + (i * K + k) % (E - ND)
            cpd[k] = (i * K + k) % ND
        out.append((cps, cpd))
    return out


def launch(kind, G, P, cps, cpd):
    if kind == "grid":
        ks._copy_rows_kernel[(P, triton.cdiv(W, ks.BLOCK))](t, cps, cpd, W, BLOCK=ks.BLOCK, num_warps=ks.WARPS)
    elif kind == "few":
        ks._copy_rows_few[(G,)](t, cps, cpd, P, W, BLOCK=ks.BLOCK, num_warps=ks.WARPS)
    else:
        ks._copy_rows_live[(G,)](t, cps, cpd, W, BS=BS, BLOCK=ks.BLOCK, num_warps=ks.WARPS)


def run(kind, G, P, K):
    pl = plans(K)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for cps, cpd in pl:  # compile and warm outside the capture
            launch(kind, G, P, cps, cpd)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, stream=s):
            for cps, cpd in pl:
                launch(kind, G, P, cps, cpd)
    gr.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(a.reps):
        gr.replay()
    e1.record()
    torch.cuda.synchronize()
    us = e0.elapsed_time(e1) * 1000 / a.reps / a.n
    bad, last = 0, {}  # the last plan to touch a device row wrote it: check those
    for cps, cpd in pl:
        for k in range(K):
            last[int(cpd[k])] = int(cps[k])
    for d, r in last.items():
        bad += int(not torch.equal(t[d], t[r]))
    for d in last:
        t[d].zero_()
    gbs = K * ROW / us / 1e3 if K else 0
    return us, gbs, bad, len(last)


cases = [("grid", 0)] + [("few", G) for G in ints(a.few)] + [("live", G) for G in ints(a.live)]
for P in ints(a.P):
    for K in ints(a.K):
        row = []
        for kind, G in cases:
            if K > P:
                continue
            us, gbs, bad, nchk = run(kind, G, P, K)
            row.append(f"{kind}{G or ''} {us:6.1f}" + (f" ({gbs:4.1f} GB/s)" if K else "") + (f" BAD {bad}/{nchk}" if bad else ""))
        print(f"P={P:3d} K={K}: " + " | ".join(row), flush=True)
mr.free()

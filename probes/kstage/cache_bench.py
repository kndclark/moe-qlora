"""Where K6's (KSTAGE=cache) decode overhead goes, on the GPU without a server.
  check    the fixed-grid copy kernel moves the same bytes as the plugin's (also on the CPU)
  idle     L layers' cache steps in one CUDA graph, every routed expert already in a slot:
           the fixed cost per layer-step (cache kernel + one copy launch per tensor)
  trace    the same graph over a routing trace at C streams; (time - idle) / copies = one
           miss's copy through UVA
  overlap  one layer's misses copied on a side stream beside stand-in compute (memory-bound
           elementwise, like decode's Marlin; a bf16 matmul, like prefill): sum or max?
Copy kernels: "grid" is the plugin's default (a program per 512 words of each possible copy,
P = min(slots, ids, E) of them, idle or not); "fewN" is KSTAGE_COPY=N, N programs that walk
the copy list.
Rows are the real expert's tensors (w13, w2 and their fp8 scales, 5.61 MB), device rows then
pinned host homes in one VMM range, as the plugin lays them out. --layers keeps host RAM down.
usage: cache_bench.py BENCH.json PROFILE.json [--layers 4] [--conc 1,16] [--slots all]
                      [--evict lru] [--steps 400] [--few 16,64,256]
       TRITON_INTERPRET=1 cache_bench.py --check-only   (the kernel check on the CPU)"""
import argparse, json, os, sys

import numpy as np
import torch
import triton

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [here, os.path.join(here, "..")]
import vllm_kstage as ks  # noqa: E402

DEV = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
# one Lightning NVFP4 expert as Marlin holds it: w13, w2, w13 scales, w2 scales
EXPERT = [((168, 3712), torch.int32), ((116, 5376), torch.int32), ((168, 1856), torch.uint8),
          ((116, 2688), torch.uint8)]


def use_copy(name):
    ks.COPY_N = 0 if name == "grid" else int(name[3:])


def check():
    """Both kernels on random copy lists, rows that do and do not fill the last block."""
    g = torch.Generator().manual_seed(7)
    for W in (1000, 1536, 77952):
        R, P = 40, 12
        t = torch.randint(-2**31, 2**31 - 1, (R, W), dtype=torch.int32, generator=g).to(DEV)
        src = torch.randperm(R, generator=g)[:2 * P]
        cps = src[:P].to(torch.int32).clone()
        cps[torch.randperm(P, generator=g)[:4]] = -1  # some entries idle
        cpd = src[P:].to(torch.int32)
        cps, cpd = cps.to(DEV), cpd.to(DEV)
        for n in (1, 3, 64):
            a, b = t.clone(), t.clone()
            ks._copy_rows_kernel[(P, triton.cdiv(W, ks.BLOCK))](a, cps, cpd, W, BLOCK=ks.BLOCK, num_warps=ks.WARPS)
            ks._copy_rows_few[(n,)](b, cps, cpd, P, W, BLOCK=ks.BLOCK, num_warps=ks.WARPS)
            assert torch.equal(a, b), (W, n)
            assert not torch.equal(a, t)
    print("check: fixed-grid copies equal the plugin's (rows of 1000, 1536, 77952 words; 1, 3, 64 programs)",
          flush=True)


def layer_rows(lay):
    """One layer's expert tensors, R = E + S rows: [0, D) device, homes pinned on the host."""
    R, out = len(lay["src"]), []
    for shape, dt in EXPERT:
        nb = int(np.prod(shape)) * torch.tensor([], dtype=dt).element_size()
        mx = ks.MixedRows(R, nb, lay["D"])
        t = mx.tensor(dt, (R,) + shape)
        t.view(R, -1)[:, :8] = torch.tensor(lay["src"], device="cuda", dtype=torch.int32)[:, None].to(dt)
        out.append((mx, ks._rows(t)))
    return out


def timed(fn, reps):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn(); torch.cuda.synchronize()
    a.record()
    for _ in range(reps):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def bench(a, lays, xs, C, K, copy):
    """ms per layer-step, all hits (idle) and over the trace; misses and copies a layer-step over
    the trace (misses past the free slots are not copied: Marlin reads them in their home row)."""
    use_copy(copy)
    L = len(lays)
    hstats, dstats = ks._stats_buffer(L)
    caches = [ks._Cache(lay, dstats[l], evict=a.evict, shift=a.shift) for l, lay in enumerate(lays)]
    for c, rows in zip(caches, a.rows):
        c.rows = [r for _, r in rows]
    static = torch.zeros(L, C * K, dtype=torch.int32, device="cuda")
    # idle ids: C*K routings over experts already in slots, the same every step
    hit = torch.tensor([[lay["owner"][j % len(lay["owner"])] for j in range(C * K)] for lay in lays],
                       dtype=torch.int32, device="cuda")
    static.copy_(hit)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for l, c in enumerate(caches):
            c.step(static[l].view(C, K))
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for l, c in enumerate(caches):
            c.step(static[l].view(C, K))

    def idle():
        static.copy_(hit); g.replay()
    t_idle = timed(idle, 200) / L
    xd = torch.from_numpy(xs).cuda()
    n = xd.shape[0]
    it = iter(range(n))

    def one():
        static.copy_(xd[next(it)]); g.replay()
    one()  # first step: compile nothing, but fills the slots from the trace's start
    torch.cuda.synchronize()
    h1 = hstats.sum(0).tolist()
    t_tr = timed(one, n - 2) / L
    h2 = hstats.sum(0).tolist()
    steps = (n - 1) * L  # the stats also count timed()'s warm-up step
    misses, copies = (h2[1] - h1[1]) / steps, (h2[2] - h1[2]) / steps
    per = (t_tr - t_idle) / copies if copies > 0 else float("nan")
    return t_idle, t_tr, misses, copies, per


def trace_ids(a, segs, L, C, K):
    """[steps, L, C*K] routed ids of C streams side by side, as cache_test.run builds them."""
    out = []
    for g in range(0, len(segs) - C + 1, C):
        grp = segs[g: g + C]
        n = min(len(r) for r in grp)
        x = np.stack([r[:n] for r in grp], 1).transpose(0, 2, 1, 3).reshape(n, L, -1)
        out.append(x)
        if sum(len(o) for o in out) >= a.steps:
            break
    return np.concatenate(out)[: a.steps].astype(np.int32)


def overlap(a):
    """One layer's k misses (w13 rows only: 2.49 MB each), alone and beside stand-in compute."""
    shape, dt = EXPERT[0]
    R, D = 64, 32
    nb = int(np.prod(shape)) * 4
    mx = ks.MixedRows(R, nb, D)
    t = ks._rows(mx.tensor(dt, (R,) + shape))
    ks._keep.append(mx)
    W = t.shape[1]
    x = torch.randn(64 << 20, device="cuda", dtype=torch.bfloat16)  # 128 MiB: read + write ~0.3 ms
    m1 = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()
    comps = {"elementwise": lambda: x.mul_(1.0), "matmul": lambda: torch.mm(m1, m1)}
    for k in (1, 4, 16):
        cps = torch.full((64,), -1, dtype=torch.int32, device="cuda")
        cps[:k] = torch.arange(D, D + k, dtype=torch.int32)
        cpd = torch.arange(64, dtype=torch.int32, device="cuda") % D
        P = 64
        for name in ["grid"] + [f"few{n}" for n in a.few]:
            if name == "grid":
                cp = lambda: ks._copy_rows_kernel[(P, triton.cdiv(W, ks.BLOCK))](t, cps, cpd, W, BLOCK=ks.BLOCK,
                                                                     num_warps=ks.WARPS)
            else:
                nprog = int(name[3:])
                cp = lambda: ks._copy_rows_few[(nprog,)](t, cps, cpd, P, W, BLOCK=ks.BLOCK, num_warps=ks.WARPS)
            tc = timed(cp, 20)
            row = [f"k={k:2d} {name:7s} copy {tc:7.3f} ms ({k * nb / tc / 1e6:5.1f} GB/s)"]
            for cn, comp in comps.items():
                tp = timed(comp, 20)

                def both():
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        cp()
                    comp()
                    torch.cuda.current_stream().wait_stream(side)
                tb = timed(both, 20)
                row.append(f"{cn} {tp:6.3f} + copy -> {tb:6.3f} (sum {tc + tp:6.3f}, max {max(tc, tp):6.3f})")
            print("  " + "; ".join(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", nargs="?"); ap.add_argument("profile", nargs="?")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--kinds", default="eval,code")
    ap.add_argument("--layers", type=int, default=4, help="first N MoE layers (host RAM: ~0.72 GB each)")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--slots", default="all")
    ap.add_argument("--conc", default="1,16")
    ap.add_argument("--cold-gb", type=float, default=4)
    ap.add_argument("--evict", default="lru")
    ap.add_argument("--shift", type=int, default=6)
    ap.add_argument("--few", default="16,64,256", help="program counts for the fixed-grid copy")
    ap.add_argument("--no-overlap", action="store_true")
    a = ap.parse_args()
    a.few = [int(v) for v in a.few.split(",")]
    check()
    if a.check_only:
        return
    import expert_cache_sim as sim
    from cache_test import cold_split
    reqs, moe, K = sim.load(a.bench)
    counts = json.load(open(a.profile))["counts"]
    E = len(counts[str(moe[0])])
    cold = cold_split(counts, moe, E, int(a.cold_gb * 2**30 // sim.E_BYTES))
    keep = list(range(a.layers))
    segs = sim.segments([r[:, keep] for k in a.kinds.split(",") for r in reqs[k]])
    moe = [moe[i] for i in keep]
    print(f"{torch.cuda.get_device_name()}: {len(moe)} layers, cold per layer {[cold[d] for d in moe]}, "
          f"BLOCK {ks.BLOCK} WARPS {ks.WARPS}", flush=True)
    for S in a.slots.split(","):
        lays = [ks._cache_layout(counts[str(d)], E - cold[d], (E - cold[d]) if S == "all"
                                 else min(int(S), E - cold[d])) for d in moe]
        a.rows = [layer_rows(lay) for lay in lays]
        for C in map(int, a.conc.split(",")):
            xs = trace_ids(a, segs, len(moe), C, K)
            for copy in ["grid"] + [f"few{n}" for n in a.few]:
                ti, tt, ms, cp, per = bench(a, lays, xs, C, K, copy)
                print(f"slots {S} C={C:2d} {copy:7s}: idle {ti * 1e3:7.1f} us/layer-step; trace "
                      f"{tt * 1e3:7.1f} us, {ms:.2f} misses {cp:.2f} copies/layer-step -> {per:.3f} ms a copy "
                      f"({sim.E_BYTES / per / 1e6:.1f} GB/s)", flush=True)
        for mx, _ in sum(a.rows, []):
            mx.free()
        a.rows = None
        torch.cuda.empty_cache()
    use_copy("grid")
    if not a.no_overlap:
        print("overlap (ms; k = misses):", flush=True)
        overlap(a)


if __name__ == "__main__":
    main()

"""KSTAGE_AHEAD_FUSE checks and timing without a server: _ahead_kernel against _Ahead.plan's
unfused path on twin K6 caches (one layer each, made-up profile counts), Lightning's gate shape.
  scores  the kernel's sigmoid(x w.T) + bias against torch's fp32 (max abs error), and how many
          tokens' top-K sets differ from torch's on torch's own scores (near ties)
  state   the unfused path fed the kernel's own top K (torch.topk of its scores) must leave the
          same where/owner/stamp/clock/freq/pf/need/cps/cpd, stats, plan, done flag and small
          rows, step after step, with a real K6 step (AH 2) on the same ids after each ahead
          assign; once eager, once with the kernel replayed from a CUDA graph
  time    us a layer in a CUDA graph of --layers layers (distinct gates and caches), unfused
          vs fused for each --beb and --hc; --flush MiB written between layers keeps the gates out of
          L2, as decode's expert reads do (its own time is measured and taken off)
usage (vLLM image): python3 /k/ahead_fused_test.py [--M 1,4,16,48] [--steps 300] [--evict lru,lfu]
                    [--slots all,16] [--layers 23] [--beb 16] [--hc 384] [--flush 128]"""
import argparse
import math
import sys

import torch

sys.path.insert(0, "/k")
import vllm_kstage as ks  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--M", default="1,4,16,48")
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--evict", default="lru,lfu")
ap.add_argument("--slots", default="all,16")
ap.add_argument("--layers", type=int, default=23)
ap.add_argument("--beb", default="16")
ap.add_argument("--hc", default="384")
ap.add_argument("--flush", type=int, default=128)
ap.add_argument("--reps", type=int, default=50)
a = ap.parse_args()
ints = lambda s: [int(v) for v in s.split(",") if v]

E, H, K, COLD = 128, 2688, ks.AHEAD_K, 25  # Lightning: 128 experts, hidden 2688; ~25 cold a layer at 3 GiB
D = E - COLD
g = torch.Generator(device="cuda").manual_seed(0)
print(f"{torch.cuda.get_device_name()}: E {E}, H {H}, top {K}, {D} device rows", flush=True)


def cache(seed, slots, evict):
    """A K6 layer from made-up counts, with small rows (int32 x 8) that name the expert they hold."""
    cnt = torch.randint(0, 10**6, (E,), generator=torch.Generator().manual_seed(seed)).tolist()
    S = D if slots == "all" else min(int(slots), D)
    lay = ks._cache_layout(cnt, D, S)
    c = ks._Cache(lay, None, evict=evict)
    c.ah = 2
    src = torch.tensor(lay["src"], dtype=torch.int32, device="cuda")
    c.rows = [ks._rows(src[:, None].expand(len(src), 8).contiguous())]
    return c


def gate(seed):
    gg = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(E, H, device="cuda", generator=gg) / math.sqrt(H)).contiguous()
    return w, torch.randn(E, device="cuda", generator=gg) * 0.1


def bufs(c):
    return (torch.zeros(1 + 2 * c.BS, dtype=torch.int64, device="cuda"),
            torch.zeros(1, dtype=torch.int32, device="cuda"), torch.zeros(6, dtype=torch.int64, device="cuda"))


def unfused(c, x, w, b, plan, done, stats, rows, ids=None):
    if ids is None:
        ids = ((x.float() @ w.t()).sigmoid() + b).topk(K, dim=-1).indices
    c.ahead(ids, stats, rows)
    ks._plan_kernel[(1,)](c.cps, c.cpd, plan, done, BS=c.BS)


def diff(A, B, pa, pb):
    for n in ("where", "owner", "stamp", "clock", "freq", "pf", "need", "cps", "cpd"):
        if not torch.equal(getattr(A, n), getattr(B, n)):
            return n
    for n, u, v in zip(("plan", "done", "stats"), pa, pb):
        k = 1 + 2 * int(u[0]) if n == "plan" else len(u)
        if not torch.equal(u[:k], v[:k]):
            return n
    for c in (A, B):  # every expert the assign routed is in a row that names it
        e = c.need.nonzero().view(-1)
        if not torch.equal(c.rows[0][c.where[e].long(), 0].long(), e):
            return "rows"
    return None


def check(slots, evict, M, graph):
    A, B = cache(1, slots, evict), cache(1, slots, evict)
    w, b = gate(1)
    pa, pb = bufs(A), bufs(B)
    sc = torch.empty(ks.fused_scratch(64, E, H), dtype=torch.float32, device="cuda")
    cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
    xs = torch.zeros(M, H, dtype=torch.bfloat16, device="cuda")
    fused = lambda: ks.fused_plan(B, xs, w, b, sc, cnt, pb[0], pb[1], pb[2], B.rows)
    if graph:
        W = cache(1, slots, evict)  # compile on a throwaway twin; capture runs nothing
        pw = bufs(W)
        ks.fused_plan(W, xs, w, b, sc, cnt, pw[0], pw[1], pw[2], W.rows)
        torch.cuda.synchronize()
        gr, st = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.graph(gr, stream=st):
            fused()
    err, flips, bad = 0.0, 0, None
    for s in range(a.steps):
        xs.copy_(torch.randn(M, H, device="cuda", generator=g).to(torch.bfloat16))
        gr.replay() if graph else fused()
        ref = (xs.float() @ w.t()).sigmoid() + b
        got = sc[:M * E].view(M, E)
        err = max(err, (got - ref).abs().max().item())
        ik = got.topk(K, dim=-1).indices
        flips += int((ref.topk(K, dim=-1).indices.sort(-1).values != ik.sort(-1).values).any(-1).sum())
        unfused(A, xs, w, b, *pa, A.rows, ids=ik)
        bad = diff(A, B, pa, pb)
        if bad:
            bad = f"{bad} at step {s}"
            break
        for p in (pa, pb):
            p[1].zero_()  # as _Ahead.wait does
        rid = torch.where(torch.rand(M, K, device="cuda", generator=g) < 0.3,
                          torch.randint(0, E, (M, K), device="cuda", generator=g), ik)
        if not torch.equal(A.step(rid), B.step(rid)):
            bad = f"real step out at step {s}"
            break
    fills = int(pb[2][2])
    print(f"  {evict} slots {slots:>3} M {M:2d} {'graph' if graph else 'eager'}: score err {err:.1e}, "
          f"top-{K} sets off torch's {flips}/{a.steps * M} tokens, {fills} fills, "
          f"{'SAME' if not bad else 'DIFF ' + bad}", flush=True)
    return not bad


def timing(M, beb, hc):
    ks.AHEAD_BEB, ks.AHEAD_HC = beb, hc
    L = a.layers
    cs = [cache(10 + l, "all", "lfu") for l in range(L)]
    gs = [gate(10 + l) for l in range(L)]
    ps = [bufs(c) for c in cs]
    sc = torch.empty(ks.fused_scratch(64, E, H), dtype=torch.float32, device="cuda")
    cnt = torch.zeros(L, dtype=torch.int32, device="cuda")
    x = torch.randn(M, H, device="cuda", generator=g).to(torch.bfloat16)
    fl = torch.empty(a.flush * 2**20 // 4, dtype=torch.float32, device="cuda") if a.flush else None
    arms = {"flush": lambda l: None,
            "unfused": lambda l: unfused(cs[l], x, *gs[l], *ps[l], []),
            "fused": lambda l: ks.fused_plan(cs[l], x, *gs[l], sc, cnt[l:], *ps[l], [])}
    out = {}
    for name, f in arms.items():
        st = torch.cuda.Stream()
        with torch.cuda.stream(st):
            for l in range(L):
                f(l)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=st):
                for l in range(L):
                    if fl is not None:
                        fl.zero_()
                    f(l)
        gr.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(a.reps):
            gr.replay()
        e1.record()
        torch.cuda.synchronize()
        out[name] = e0.elapsed_time(e1) * 1000 / a.reps / L
    u, f = out["unfused"] - out["flush"], out["fused"] - out["flush"]
    print(f"  M {M:2d} BEB {beb:3d} HC {hc:4d}: unfused {u:5.1f} us a layer, fused {f:5.1f} ({f - u:+.1f}); "
          f"flush {out['flush']:.1f} us a layer taken off", flush=True)


ok = True
print("state (twin caches, the unfused path fed the kernel's top K):", flush=True)
for evict in a.evict.split(","):
    for slots in a.slots.split(","):
        for M in ints(a.M):
            for graph in (False, True):
                ok &= check(slots, evict, M, graph)
print(f"time (CUDA graph of {a.layers} layers, {a.flush} MiB flush between, mean of {a.reps} replays):", flush=True)
for M in ints(a.M):
    for beb in ints(a.beb):
        for hc in ints(a.hc):
            timing(M, beb, hc)
print("ALL SAME" if ok else "SOME DIFF")
sys.exit(0 if ok else 1)

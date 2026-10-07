"""K6 (KSTAGE=cache) checks without a server, on a recorded routing trace.
  logic  the cache kernel against a Python LRU (or decayed LFU), step by step, and its total
         misses against expert_cache_sim.py's lru_bytes for the same cache (they must agree exactly)
  rows   after each step's copies every routed id's row holds that expert's bytes
         (int32, int16 and uint8 rows; on the GPU one layer's rows are a VMM MixedRows)
  stats  (GPU) the host-mapped counters equal the reference's misses and copies
  graph  (GPU) a captured step replays with new ids exactly as an uncaptured twin
  ahead  (--ahead) before each step an ahead assign (KSTAGE_AHEAD) on a made-up prediction (each
         token's last two ids taken from the next step), against the reference run the same
         way, and the fill-ahead tally (useful, polluted, wasted) against the reference's
usage: cache_test.py BENCH.json PROFILE.json [--kinds eval,code] [--layers N] [--steps N]
                     [--slots all,16] [--conc 1,4,16] [--cold-gb 4] [--evict lru,lfu] [--shift 6] [--ahead]
TRITON_INTERPRET=1 runs the kernels on the CPU (torch tensors on the CPU)."""
import argparse, json, os, sys, time

import numpy as np
import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [here, os.path.join(here, "..")]
import vllm_kstage as ks  # noqa: E402
import expert_cache_sim as sim  # noqa: E402

DEV = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


class Ref:
    """lru_bytes for one layer as sets: the pinned experts, and each slot expert's last use
    (lfu: every expert's decayed use count instead)."""

    def __init__(self, lay, evict="lru", shift=6):
        self.pin = {e for e, h in enumerate(lay["home"]) if h < 0}
        self.last = dict(zip(lay["owner"], lay["stamp"]))
        self.t = 0
        self.lfu, self.shift, self.freq = evict == "lfu", shift, list(lay["freq"])
        self.pf, self.tally = {}, [0, 0, 0, 0]  # ahead: useful, polluted, wasted (real steps, ahead assigns)

    def step(self, ids, ahead=False):
        touched = set(ids)
        miss = sorted(e for e in touched if e not in self.pin and e not in self.last)
        for e in touched & self.last.keys():
            self.last[e] = self.t
        if self.lfu and not ahead:
            self.freq = [f - (f >> self.shift) + (1 << 20) * (e in touched) for e, f in enumerate(self.freq)]
        score = {e: self.freq[e] for e in self.last} if self.lfu else self.last
        cand = sorted((score[e], e) for e in self.last if e not in touched)
        na = min(len(miss), len(cand))
        out = [e for _, e in cand[:na]]
        if ahead:
            self.tally[3] += sum(self.pf.get(e) == 1 for e in out)
            self.pf.update({e: 2 for e in out} | {e: 1 for e in miss[:na]})
        else:
            self.tally[0] += sum(self.pf.get(e) == 1 for e in touched if e not in miss)
            self.tally[1] += sum(self.pf.get(e) == 2 for e in miss)
            self.pf = {e: v for e, v in self.pf.items() if v == 1 and e not in touched}
            self.tally[2] += sum(self.pf.pop(e, 0) == 1 for e in out)
        for e in out:
            del self.last[e]
        for e in miss[:na]:
            self.last[e] = self.t
        self.t += 1
        return len(miss), na


def cold_split(counts, moe, E, n_cold):
    """vllm_kstage._cold_split, policy global."""
    pairs = sorted((counts[str(d)][e], -d, -e, d) for d in moe for e in range(E))
    cold = dict.fromkeys(moe, 0)
    for *_, d in pairs[:n_cold]:
        cold[d] += 1
    return cold


def row_tensors(lay, E, mixed):
    """Rows that name the expert they hold: int32 x 8, int16 x 3, uint8 x 3 (ids < 128)."""
    src = torch.tensor(lay["src"], device=DEV)
    R = len(lay["src"])
    out = []
    for dt, w in ((torch.int32, 8), (torch.int16, 3), (torch.uint8, 3)):
        t = src[:, None].to(dt).expand(R, w).contiguous()
        if mixed and dt == torch.int32:  # device rows [0, D), host rows after, as the plugin lays them out
            mx = ks.MixedRows(R, w * 4, lay["D"])
            m = mx.tensor(torch.int32, (R, w))
            m.copy_(t)
            ks._keep.append(mx)
            t = m
        out.append(t)
    return out


def run(a, segs, moe, K, counts, E, cold, C, slots, mixed_first, evict):
    L = len(moe)
    lays = [ks._cache_layout(counts[str(d)], E - cold[d], (E - cold[d]) if slots == "all"
                             else min(int(slots), E - cold[d])) for d in moe]
    nf = 6 if a.ahead else 3
    buf = lambda: ks._stats_buffer(L, nf) if DEV == "cuda" else (lambda t: (t, t))(torch.zeros(L, nf, dtype=torch.int64))
    (hstats, stats), (hahead, sahead) = buf(), buf()
    caches = [ks._Cache(lay, stats[l], device=DEV, evict=evict, shift=a.shift) for l, lay in enumerate(lays)]
    for c in caches:
        c.ah = 2 if a.ahead else 0
    rows = []
    for l, (c, lay) in enumerate(zip(caches, lays)):
        c.rows = [ks._rows(t) for t in row_tensors(lay, E, mixed_first and l == 0)]
        rows.append(c.rows)
    refs = [Ref(lay, evict, a.shift) for lay in lays]
    bad = torch.zeros(1, dtype=torch.int64, device=DEV)
    miss_r = copy_r = miss_a = copy_a = steps = 0
    t0 = time.perf_counter()
    for g in range(0, len(segs) - C + 1, C):
        grp = segs[g: g + C]
        n = min(len(r) for r in grp)
        x = np.stack([r[:n] for r in grp], 1).transpose(0, 2, 1, 3).reshape(n, L, -1)  # as lru_bytes
        xd = torch.from_numpy(x.astype(np.int32)).to(DEV)
        for s in range(n):
            if a.steps and steps >= a.steps:
                break
            for l in range(L):
                ids = xd[s, l].view(C, K)
                if a.ahead:
                    pred = ids.clone()
                    pred[:, -2:] = xd[min(s + 1, n - 1), l].view(C, K)[:, -2:]
                    caches[l].ahead(pred, sahead[l], caches[l].rows)
                    am, ac = refs[l].step(pred.view(-1).tolist(), ahead=True)
                    miss_a += am; copy_a += ac
                out = caches[l].step(ids).view(-1).long()
                for t in rows[l]:
                    bad += (t[out, 0].long() != ids.view(-1).long()).sum()
                nm, na = refs[l].step(x[s, l].tolist())
                miss_r += nm; copy_r += na
            steps += 1
            if DEV == "cpu" or steps % 256 == 0:  # the cached sets agree
                for l in range(L):
                    got = set(caches[l].owner[:caches[l].S].tolist()) if caches[l].S else set()
                    assert got == set(refs[l].last), f"C={C} S={slots} layer {moe[l]} step {steps}: cache sets differ"
                    if evict == "lfu":
                        assert caches[l].freq.tolist() == refs[l].freq, f"C={C} S={slots} step {steps}: counts differ"
        else:
            continue
        break
    if DEV == "cuda":
        torch.cuda.synchronize()
    assert int(bad) == 0, f"C={C} S={slots}: {int(bad)} routed ids read another expert's row"
    msg = f"{evict} C={C:2} slots={slots:>4}: {steps} steps x {L} layers, ref misses {miss_r} copies {copy_r}"
    v, want = hstats.sum(0).tolist(), [steps * L, miss_r, copy_r]
    if a.ahead:
        t = [sum(r.tally[i] for r in refs) for i in range(4)]
        want += t[:3]
        va = hahead.sum(0).tolist()
        assert va == [steps * L, miss_a, copy_a, 0, 0, t[3]], f"ahead stats {va}: {miss_a} {copy_a} {t}"
        msg += f", ahead fills {va[2]}: used {t[0]}, missed after an ahead eviction {t[1]}, evicted unused {t[2] + t[3]}"
    assert v == want, f"stats {v} != {want}"
    msg += ", stats equal"
    if not a.steps and not a.ahead:  # the whole trace: the simulator's total too
        hot = np.zeros((L, E), bool)
        for l, lay in enumerate(lays):
            hot[l, lay["src"][:lay["D"]]] = True
        prof = np.array([counts[str(d)] for d in moe], np.int64)
        b = sim.lru_bytes(segs, hot, prof, C, None if slots == "all" else int(slots), evict, a.shift)
        miss_s = round(b * steps / sim.E_BYTES)
        assert miss_s == miss_r, f"simulator {miss_s} misses != reference {miss_r}"
        msg += f", simulator {miss_s} equal ({b / 1e6:.1f} MB a step)"
    print(msg + f"; rows ok; {time.perf_counter() - t0:.1f}s", flush=True)


def graph_test(lay, K, xs, evict="lru", shift=6):
    """Capture one step; replay it on new ids; an eager twin must end in the same state."""
    A = ks._Cache(lay, device="cuda", evict=evict, shift=shift)
    B = ks._Cache(lay, device="cuda", evict=evict, shift=shift)
    for c in (A, B):
        c.rows = [ks._rows(t) for t in row_tensors(lay, len(lay["where"]), False)]
    static = torch.from_numpy(xs[0].astype(np.int32)).cuda().view(-1, K)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):  # warm up (compiles) on both, so the twins stay equal
        A.step(static); B.step(static)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outB = B.step(static)
    for x in xs[1:]:
        ids = torch.from_numpy(x.astype(np.int32)).cuda().view(-1, K)
        outA = A.step(ids)
        static.copy_(ids)
        g.replay()
        assert torch.equal(outA, outB), "graph replay remapped differently"
    for f in ("where", "owner", "stamp", "clock", "freq"):
        assert torch.equal(getattr(A, f), getattr(B, f)), f"graph replay: {f} differs"
    for ra, rb in zip(A.rows, B.rows):
        assert torch.equal(ra, rb), "graph replay: rows differ"
    print(f"graph ({evict}): {len(xs) - 1} replays of a captured step match the eager twin", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench"); ap.add_argument("profile")
    ap.add_argument("--kinds", default="eval,code")
    ap.add_argument("--layers", type=int, default=0, help="first N MoE layers only")
    ap.add_argument("--steps", type=int, default=0, help="stop after N steps (skips the simulator total)")
    ap.add_argument("--slots", default="all,16")
    ap.add_argument("--conc", default="1,4,16")
    ap.add_argument("--cold-gb", type=float, default=4)
    ap.add_argument("--evict", default="lru", help="lru, lfu or both (lru,lfu)")
    ap.add_argument("--shift", type=int, default=6, help="lfu: counts lose count >> shift a step")
    ap.add_argument("--ahead", action="store_true", help="an ahead assign before every step, and its tally")
    a = ap.parse_args()
    reqs, moe, K = sim.load(a.bench)
    counts = json.load(open(a.profile))["counts"]
    E = len(counts[str(moe[0])])
    cold = cold_split(counts, moe, E, int(a.cold_gb * 2**30 // sim.E_BYTES))
    keep = list(range(a.layers or len(moe)))
    segs = sim.segments([r[:, keep] for k in a.kinds.split(",") for r in reqs[k]])
    moe = [moe[i] for i in keep]
    print(f"{DEV}: {len(moe)} layers, {sum(len(s) for s in segs)} tokens ({a.kinds}), "
          f"cold per layer {[cold[d] for d in moe]}", flush=True)
    first = True
    for ev in a.evict.split(","):
        for C in map(int, a.conc.split(",")):
            for S in a.slots.split(","):
                run(a, segs, moe, K, counts, E, cold, C, S, DEV == "cuda" and first, ev)
                first = False
    if DEV == "cuda":
        d = moe[0]
        lay = ks._cache_layout(counts[str(d)], E - cold[d], min(16, E - cold[d]))
        xs = [s[t, 0] for s in segs[:64] for t in range(0, len(s), 7)][:200]
        for ev in a.evict.split(","):
            graph_test(lay, K, xs, ev, a.shift)
    print("all checks passed")


if __name__ == "__main__":
    main()

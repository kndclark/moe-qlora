"""Experts in RAM, option J: replay Lightning's real routings through expert placements.

Reads bench-TAG.json and its .experts-KIND.bin files (offload_bench.py experts mode:
[tokens, decoder layers, top_k] uint8 per request, "lens" to cut requests apart; only the
MoE layers' rows carry routings). For each
offload budget it compares where the cold experts live:

  layers   what --cpu-offload-gb does now: whole MoE layers, first layers first
  static   the least-routed experts of every layer, same count per layer (hot/cold, K)
  global   the least-routed (layer, expert) pairs anywhere
  lru      a per-layer VRAM cache holding as many experts as the global policy keeps hot,
           warm-started with that hot set and kept across requests (as a server would);
           each step's touched cold experts are copied in, evicting the least recently used;
           --slots S pins all but S of the cache to the profile's hottest experts (the rest
           need no RAM copy) and lets only S slots a layer turn over (K6 hybrid)

Placements are learned on one kind of text and scored on the other (eval prompts vs code),
on the held-out half of the same kind (learned on even requests, scored on odd), and
in-sample, so skew that does not transfer shows up as a gap. Requests are cut into 256-token
segments before they are grouped C at a time, so 8 long code requests still fill c=16. A decode step at
concurrency C reads each distinct cold expert its C tokens route to once (UVA); a 2,048-token
prefill batch reads every cold expert it touches once if copied in (K2), or once per Marlin
block of 64 tokens routed to it when Marlin reads it through UVA (now). Extra time = bytes / --gbs, the GPU's
read rate from host memory: fit it to a measured row with --fit before trusting the rest.

Run where numpy exists, e.g. inside the vLLM image (CPU only):
  docker run --rm --pull never -v $PWD:/w -w /w --entrypoint python3 vllm/vllm-openai:v0.29.0 \\
    probes/expert_cache_sim.py results/kv-levers/bench-p7-experts.json
"""
import argparse, json
import numpy as np

E_BYTES = 2 * 1856 * 2688 * 9 // 16  # one expert: NVFP4 up + down projections, fp8 scale per 16


def load(path):
    """Routings per kind, cut into requests and cut down to the MoE layers. vLLM returns a
    row for every decoder layer; the Mamba and attention layers' rows carry no routing (at
    most top_k distinct ids), and counting them made the median layer look fully skewed."""
    d = json.load(open(path))["experts"]
    L, K = d["shape_layers_topk"]
    reqs = {}
    for kind, lens in d["lens"].items():
        raw = np.fromfile(f"{path}.experts-{kind}.bin", dtype=np.uint8)  # offload_bench.py's naming
        a, o = [], 0
        for n, _prompt in lens:
            a.append(raw[o: o + n * L * K].reshape(n, L, K)); o += n * L * K
        assert o == raw.size, f"{kind}: lens cover {o} of {raw.size} bytes"
        reqs[kind] = a
    allr = np.concatenate([r for a in reqs.values() for r in a])
    moe = [l for l in range(L) if np.unique(allr[:, l]).size > K]
    reqs = {k: [r[:, moe] for r in a] for k, a in reqs.items()}
    return reqs, moe, K


def freq(reqs, L, nE):
    c = np.zeros((L, nE), np.int64)
    for r in reqs:
        for l in range(L):
            c[l] += np.bincount(r[:, l].ravel(), minlength=nE)
    return c


def placement(policy, cnt, n_cold, L, nE):
    """Boolean [L, nE]: True = in RAM."""
    cold = np.zeros((L, nE), bool)
    if policy == "layers":
        full, rest = divmod(n_cold, nE)
        cold[:full] = True
        if rest:
            cold[full, :rest] = True  # per-parameter cut: the next layer's first experts
    elif policy == "static":
        per = n_cold // L
        for l in range(L):
            cold[l, np.argsort(cnt[l], kind="stable")[:per]] = True
        extra = n_cold - per * L  # remainder: globally least-routed hot ones
        if extra:
            idx = np.argsort(np.where(cold, np.iinfo(np.int64).max, cnt), axis=None, kind="stable")[:extra]
            cold.flat[idx] = True
    elif policy == "global":
        cold.flat[np.argsort(cnt, axis=None, kind="stable")[:n_cold]] = True
    return cold


def segments(reqs, n=256):
    """Requests cut into pieces of at most n tokens: the streams grouped C at a time."""
    return [r[i: i + n] for r in reqs for i in range(0, len(r), n)]


def decode_bytes(reqs, cold, C, L):
    """Mean bytes read from RAM per decode step with C requests stepping together."""
    tot, steps = 0, 0
    for g in range(0, len(reqs) - C + 1, C):
        grp = reqs[g: g + C]
        n = min(len(r) for r in grp)
        x = np.stack([r[:n] for r in grp], 1)  # [n, C, L, K]
        for l in range(L):
            m = np.zeros((n, cold.shape[1]), bool)
            np.put_along_axis(m, x[:, :, l, :].reshape(n, -1).astype(np.int64), True, axis=1)
            tot += int((m & cold[l]).sum())
        steps += n
    return tot * E_BYTES / max(steps, 1)


def marlin_block_m(M, K, nE):
    """Marlin MoE's block_size_m (fused_moe/experts/marlin_moe.py): the first size whose
    tokens-per-expert ratio is under 0.9, else 64. Each block of up to this many tokens
    routed to an expert reads that expert's weights again."""
    for bs in (8, 16, 32, 48, 64):
        if M * K / nE / bs < 0.9:
            return bs
    return 64


def prefill_bytes(reqs, cold, L, K, batch=2048):
    """Mean bytes read from RAM per prefill batch of `batch` tokens: (copy-once, Marlin UVA).
    Copy-once reads each touched cold expert once (paged experts, K2); Marlin reading through
    UVA reads it once per block of block_size_m tokens routed to it."""
    flat = np.concatenate(reqs)
    bs = marlin_block_m(batch, K, cold.shape[1])
    once, uva, nb = 0, 0, 0
    for s in range(0, len(flat) - batch + 1, batch):
        x = flat[s: s + batch]
        for l in range(L):
            n = np.bincount(x[:, l].ravel(), minlength=cold.shape[1])
            once += int(((n > 0) & cold[l]).sum())
            uva += int((np.ceil(n / bs) * cold[l]).sum())
        nb += 1
    return once * E_BYTES / max(nb, 1), uva * E_BYTES / max(nb, 1)


def lru_bytes(segs, hot, prof, C, slots=None, policy="lru", shift=6, admit=False, bypass=1.0):
    """Mean bytes copied in per decode step, C segments stepping together, with a per-layer
    cache of hot.sum(1) experts that starts as `hot` (least-routed in `prof` evicted first)
    and persists across requests. A step reads each touched expert not in the cache once,
    the same bytes a UVA read costs, and keeps it, evicting experts this step did not touch.
    slots: only this many of a layer's cache entries are ever evicted; the rest stay pinned
    to the experts `prof` routes to most.
    policy: which unpinned entry goes first. "lru" the least recently used; "lru2" the one whose
    second-last use is oldest (LRU-K with K=2: experts used once leave before ones used twice);
    "lfu" the fewest uses, decayed: each step every count loses count >> shift, each use adds
    2**20 (integers, as the K6 kernel; half-life about 0.69 * 2**shift steps).
    admit: TinyLFU-style admission. The i-th most-used miss replaces the i-th victim only if
    its score is higher; the others are read in place, at `bypass` times a copy's cost
    (copy rate / UVA read rate)."""
    L, nE = hot.shape
    inc, cap, big = hot.copy(), hot.sum(1), np.iinfo(np.int64).max
    last = np.argsort(np.argsort(prof, 1, kind="stable"), 1).astype(np.int64) - nE  # all before step 1
    prev = last - nE                                   # lru2: the use before `last`, also in profile order
    cnt = last + nE + 1                                # lfu: the profile's order, below one use (2**20)
    pin = np.zeros_like(inc)
    if slots is not None:
        pin = inc & (last >= (nE - np.maximum(cap - slots, 0))[:, None] - nE)  # the cap-S most routed
    rows, ar = np.arange(L)[:, None], np.arange(nE)
    tot, steps, t, by = 0, 0, 1, 0
    for g in range(0, len(segs) - C + 1, C):
        grp = segs[g: g + C]
        n = min(len(r) for r in grp)
        x = np.stack([r[:n] for r in grp], 1).transpose(0, 2, 1, 3).reshape(n, L, -1)  # [n, L, C*K]
        for s in range(n):
            touched = np.zeros((L, nE), bool)
            touched[rows, x[s]] = True
            miss = touched & ~inc
            nm = miss.sum(1)
            tot += int(nm.sum())
            if policy == "lru2":
                prev[touched] = last[touched]
            elif policy == "lfu":
                cnt -= cnt >> shift
                cnt[touched] += 1 << 20
            last[touched] = t
            if nm.any():
                score = {"lru": last, "lru2": prev, "lfu": cnt}[policy]
                key = np.where(inc & ~touched & ~pin, score, big)
                rank = np.empty_like(key)
                order = np.argsort(key, 1, kind="stable")
                rank[rows, order] = ar
                na, enter = nm, miss
                if admit:  # scores fall along the misses and rise along the victims: a prefix enters
                    mrank = np.empty((L, nE), np.int64)
                    mrank[rows, np.argsort(np.where(miss, -score, np.inf), 1, kind="stable")] = ar
                    victim = np.take_along_axis(key, order, 1)[rows, np.minimum(mrank, nE - 1)]
                    enter = miss & (score > victim) & (victim != big)
                    na = enter.sum(1)
                    by += int((nm - na).sum())
                inc &= ~((rank < na[:, None]) & (key != big))
                free = cap - inc.sum(1)  # less than nm when the cache is smaller than the step's set
                inc |= enter & (np.cumsum(enter, 1) <= free[:, None])
            t += 1
        steps += n
    return (tot + (bypass - 1) * by) * E_BYTES / max(steps, 1)


def belady_bytes(segs, hot, prof, C, slots=None):
    """lru_bytes' cache, steps and pinning, but each miss evicts the expert used again furthest
    ahead (Belady's rule, which needs the future): the fewest copies any policy of this size,
    prediction included, could make while each step holds its own experts."""
    L, nE = hot.shape
    xs = []
    for g in range(0, len(segs) - C + 1, C):
        grp = segs[g: g + C]
        n = min(len(r) for r in grp)
        xs.append(np.stack([r[:n] for r in grp], 1).transpose(0, 2, 1, 3).reshape(n, L, -1))
    x = np.concatenate(xs).astype(np.int64)  # [steps, L, C*K]
    T, never = len(x), np.int64(1) << 40
    rows, ar = np.arange(L)[:, None], np.arange(nE)
    nxt, seen = np.empty_like(x), np.full((L, nE), never)
    for t in range(T - 1, -1, -1):  # each routing's next use of the same expert
        nxt[t] = seen[rows, x[t]]
        seen[rows, x[t]] = t
    nu = seen  # now each expert's first use
    inc, cap, big = hot.copy(), hot.sum(1), np.iinfo(np.int64).max
    pin = np.zeros_like(inc)
    if slots is not None:
        last = np.argsort(np.argsort(prof, 1, kind="stable"), 1).astype(np.int64) - nE
        pin = inc & (last >= (nE - np.maximum(cap - slots, 0))[:, None] - nE)
    tot = 0
    for t in range(T):
        touched = np.zeros((L, nE), bool)
        touched[rows, x[t]] = True
        nu[rows, x[t]] = nxt[t]
        miss = touched & ~inc
        nm = miss.sum(1)
        tot += int(nm.sum())
        if nm.any():
            key = np.where(inc & ~touched & ~pin, -nu, big)
            rank = np.empty_like(key)
            rank[rows, np.argsort(key, 1, kind="stable")] = ar
            inc &= ~((rank < nm[:, None]) & (key != big))
            free = cap - inc.sum(1)
            inc |= miss & (np.cumsum(miss, 1) <= free[:, None])
    return tot * E_BYTES / max(T, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--budgets", default="2,4,8,12,16", help="GiB of experts in RAM")
    ap.add_argument("--conc", default="1,4,16")
    ap.add_argument("--gbs", type=float, default=24.0, help="GPU read rate from host memory, GB/s")
    ap.add_argument("--fit", default="", help="GIB:C:MS_EXTRA measured, e.g. 4:1:8.3 sets --gbs from the layers policy")
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--no-lru", action="store_true")
    ap.add_argument("--belady", action="store_true", help="also the offline bound under each lru line")
    ap.add_argument("--policies", default="", help="also these online policies under each lru line, e.g. "
                    "lru2,lfu:6 (lfu:SHIFT, half-life ~0.69 * 2**SHIFT steps); a + adds admission (lfu+:6)")
    ap.add_argument("--bypass", type=float, default=1.35, help="admission: an in-place read's cost per byte over "
                    "a copy's (laptop Triton copy 36.8 / UVA 27.2 GB/s; desktop 12.2 / 7.2 = 1.69)")
    ap.add_argument("--slots", default="", help="also score K6 hybrids: S evictable slots a layer, e.g. 8,16,32")
    ap.add_argument("--emit-profile", action="append", default=[], metavar="KIND:PATH",
                    help="write routing counts per decoder layer for vllm_kstage.py (KSTAGE_PROFILE); "
                         "KIND is a kind, its half KINDA, or all")
    a = ap.parse_args()
    reqs, moe, K = load(a.bench)
    L, nE, kinds = len(moe), a.experts, sorted(reqs)
    sets = {}
    for k in kinds:
        sets[k], sets[k + "A"], sets[k + "B"] = reqs[k], reqs[k][0::2], reqs[k][1::2]
    cnt = {k: freq(v, L, nE) for k, v in sets.items()}
    segs = {k: segments(v) for k, v in sets.items()}
    for spec in a.emit_profile:
        kind, path = spec.split(":", 1)
        c = sum(cnt[k] for k in kinds) if kind == "all" else cnt[kind]
        json.dump({"kind": kind, "source": a.bench, "counts": {str(d): c[l].tolist() for l, d in enumerate(moe)}},
                  open(path, "w"))
        print(f"profile {kind} -> {path}")
    pairs = ([(o, k) for k in kinds for o in kinds if o != k] + [(k + "A", k + "B") for k in kinds]
             + [(k, k) for k in kinds])  # cross-kind, held-out half, in-sample
    print(f"{L} MoE layers (decoder layers {moe[0]}..{moe[-1]}), top-{K}, {nE} experts, {E_BYTES / 1e6:.2f} MB each; "
          + ", ".join(f"{k}: {len(reqs[k])} requests, {sum(len(r) for r in reqs[k])} tokens" for k in kinds))
    for k in kinds:  # skew: share of routings to each layer's most-routed experts
        s = np.sort(cnt[k], 1)[:, ::-1]; s = s / s.sum(1, keepdims=True)
        print(f"skew {k}: median layer's top 16/32/64/96 experts take "
              + "/".join(f"{np.median(s[:, :m].sum(1)):.2f}" for m in (16, 32, 64, 96))
              + f" of routings (uniform: {16/nE:.2f}/{32/nE:.2f}/{64/nE:.2f}/{96/nE:.2f})")
    if a.fit:
        g, c, ms = a.fit.split(":")
        n_cold = int(float(g) * 2**30 // E_BYTES)
        b = np.mean([decode_bytes(segs[k], placement("layers", None, n_cold, L, nE), int(c), L) for k in kinds])
        a.gbs = b / (float(ms) / 1e3) / 1e9
        print(f"fit: layers policy at {g} GiB, c={c}: {b / 1e6:.0f} MB per step over {ms} ms -> {a.gbs:.1f} GB/s")
    concs = [int(x) for x in a.conc.split(",")]
    print(f"\nextra ms per decode step (bytes from RAM / {a.gbs:.1f} GB/s); prefill: GB read per 2,048-token batch,"
          " each touched expert once (K2) and as Marlin re-reads it through UVA (now)")
    for g in (float(x) for x in a.budgets.split(",")):
        n_cold = min(int(g * 2**30 // E_BYTES), L * nE)
        print(f"\n{g:g} GiB in RAM = {n_cold} experts ({n_cold / L:.1f} a layer)")
        for pol in ("layers", "static", "global"):
            for prof, test in ([(None, k) for k in kinds] if pol == "layers" else pairs):
                cold = placement(pol, cnt[prof] if prof else None, n_cold, L, nE)
                share = np.mean([(cnt[test] * cold).sum() / cnt[test].sum()])
                ms = [decode_bytes(segs[test], cold, c, L) / (a.gbs * 1e9) * 1e3 for c in concs]
                pb, pu = (x / 1e9 for x in prefill_bytes(sets[test], cold, L, K))
                tag = f"{pol:6} " + (f"learned on {prof:5} " if prof else " " * 16) + f"scored on {test:5}"
                print(f"  {tag}: {share:.3f} of routings cold; decode +"
                      + " / ".join(f"{m:.1f}" for m in ms) + f" ms at c={a.conc}; prefill GB/batch {pb:.2f} once, {pu:.2f} Marlin UVA")
        def policy_lines(sg, hot, pc, S, tag, prof, test):
            for spec in (x for x in a.policies.split(",") if x):
                pol, _, h = spec.partition(":")
                ms = [lru_bytes(sg, hot, pc, c, S, pol.rstrip("+"), int(h or 6), pol.endswith("+"), a.bypass)
                      / (a.gbs * 1e9) * 1e3 for c in concs]
                print(f"  {pol + h + (f'/{tag}' if tag != '' else ''):6} warm from {prof:5} scored on {test:5}: decode +"
                      + " / ".join(f"{m:.1f}" for m in ms) + f" ms at c={a.conc}")

        if not a.no_lru and n_cold < L * nE:
            for prof, test in pairs[:-len(kinds)]:  # in-sample adds nothing for a cache
                hot = ~placement("global", cnt[prof], n_cold, L, nE)
                ms = [lru_bytes(segs[test], hot, cnt[prof], c) / (a.gbs * 1e9) * 1e3 for c in concs]
                print(f"  lru    warm from {prof:5} scored on {test:5}: decode +"
                      + " / ".join(f"{m:.1f}" for m in ms) + f" ms at c={a.conc} (bytes copied in at the read rate)")
                if a.belady:
                    ms = [belady_bytes(segs[test], hot, cnt[prof], c) / (a.gbs * 1e9) * 1e3 for c in concs]
                    print(f"  belady warm from {prof:5} scored on {test:5}: decode +"
                          + " / ".join(f"{m:.1f}" for m in ms) + f" ms at c={a.conc} (offline bound)")
                policy_lines(segs[test], hot, cnt[prof], None, "", prof, test)
                for S in (int(x) for x in a.slots.split(",") if x):
                    ms = [lru_bytes(segs[test], hot, cnt[prof], c, S) / (a.gbs * 1e9) * 1e3 for c in concs]
                    ram = (n_cold + L * S) * E_BYTES / 2**30
                    print(f"  lru{S:<3} warm from {prof:5} scored on {test:5}: decode +"
                          + " / ".join(f"{m:.1f}" for m in ms) + f" ms at c={a.conc} ({ram:.1f} GiB of RAM copies)")
                    if a.belady:
                        ms = [belady_bytes(segs[test], hot, cnt[prof], c, S) / (a.gbs * 1e9) * 1e3 for c in concs]
                        print(f"  bel{S:<3} warm from {prof:5} scored on {test:5}: decode +"
                              + " / ".join(f"{m:.1f}" for m in ms) + f" ms at c={a.conc} (offline bound)")
                    policy_lines(segs[test], hot, cnt[prof], S, S, prof, test)


if __name__ == "__main__":
    main()

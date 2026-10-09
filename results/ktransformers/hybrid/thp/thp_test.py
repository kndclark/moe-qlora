"""KSTAGE_THP correctness: K6 steps copying misses from a THP mirror of the host rows (whose
originals are then overwritten with -1, so a read from them shows) against the stock live copy,
same layout and routing. After every step the device rows [0, D) and the remapped ids must be
identical. Rows on a VMM MixedRows as in _install_cache; int32, int16 and uint8 rows."""
import os, sys, torch
sys.path.insert(0, "/k")
import vllm_kstage as ks

assert ks.THP and ks.COPY_LIVE, "run with KSTAGE_THP=1 KSTAGE_COPY_LIVE=82"
g = torch.Generator().manual_seed(1)
E, D, K = 128, 95, 6
cnt = torch.randint(0, 1000, (E,), generator=g).tolist()
lay = ks._cache_layout(cnt, D, D)
R = E + lay["S"]


def rows(kind):
    out = []
    for dt, rb in ((torch.int32, 1 << 20), (torch.int16, 3 * 4096 + 2), (torch.uint8, 4097)):
        if kind == "mixed":
            mx = ks.MixedRows(R, rb, D)
            t = mx.tensor(torch.uint8, (R, rb)); keep.append(mx)
        else:
            t = torch.empty(R, rb, dtype=torch.uint8, device="cuda")
        t.copy_(base[dt][:R])
        out.append(ks._rows(t))
    return out


keep = []
base = {dt: torch.randint(0, 256, (R, rb), generator=g, dtype=torch.uint8).cuda()
        for dt, rb in ((torch.int32, 1 << 20), (torch.int16, 3 * 4096 + 2), (torch.uint8, 4097))}
A = ks._Cache(lay, device="cuda", evict="lfu")
B = ks._Cache(lay, device="cuda", evict="lfu")
A.rows, B.rows = rows("mixed"), rows("mixed")
print("row dtypes", [r.dtype for r in A.rows], "D", D, "R", R, "S", lay["S"])
huge = 0
for r in A.rows:
    v, k, h = ks._thp_mirror(r[D:R])
    keep.append(k); huge += h
    assert torch.equal(v, r[D:R])
    A.src[r.data_ptr()] = v
    r[D:R] = -1 if r.dtype != torch.uint8 else 255  # the mirror is now the only good copy
print(f"mirror on huge pages: {huge / 2**20:.0f} MiB")
torch.cuda.synchronize()
steps = misses = 0
for M in (1, 4, 16, 1, 16, 4) * 40:
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(M)]).to(torch.int32).cuda()
    oa, ob = A.step(ids), B.step(ids)
    torch.cuda.synchronize()
    assert torch.equal(oa, ob), "remapped ids differ"
    for ra, rb_ in zip(A.rows, B.rows):
        assert torch.equal(ra[:D], rb_[:D]), f"device rows differ at step {steps}, {ra.dtype}"
    misses += int((A.cps >= 0).sum())
    steps += 1
print(f"THP copy PASS: {steps} steps, {misses} copies, device rows and ids identical")

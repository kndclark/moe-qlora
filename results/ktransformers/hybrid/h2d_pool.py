"""Expert-sized H2D copy rate vs pinned-pool size, for torch pinned memory (4 KiB pages) and for an
madvise(MADV_HUGEPAGE) buffer registered with cudaHostRegister (2 MiB pages if THP backs it).
The GPU sits behind a translating IOMMU (DMA-FQ), so pool size may matter. Best of 3 x 1 s."""
import ctypes, json, mmap, time, torch
MiB = 2**20
EXPERT = 5_611_520  # 5.35 MiB rounded up to 4 KiB; odd sizes misalign every copy
TOTAL = 4 * 2**30


def anon_huge_kib():
    for line in open("/proc/meminfo"):
        if line.startswith("AnonHugePages:"):
            return int(line.split()[1])


def torch_pin():
    t = torch.empty(TOTAL, dtype=torch.uint8, pin_memory=True)
    t.fill_(1)
    return t, None


def thp_register():
    before = anon_huge_kib()
    m = mmap.mmap(-1, TOTAL + 2 * MiB, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    base = ctypes.addressof(ctypes.c_char.from_buffer(m))
    off = (-base) % (2 * MiB)  # 2 MiB-align the start so THP can back every page
    m.madvise(mmap.MADV_HUGEPAGE, off, TOTAL)
    t = torch.frombuffer(m, dtype=torch.uint8, count=TOTAL, offset=off)
    t.fill_(1)
    huge = anon_huge_kib() - before
    rc = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), TOTAL, 0)
    assert int(rc) == 0, f"cudaHostRegister rc={rc}"
    return t, {"thp_gib": round(huge / 2**20, 2), "registered": t.is_pinned()}


dev = torch.empty(64 * EXPERT, dtype=torch.uint8, device="cuda")
dv = [dev[j * EXPERT:(j + 1) * EXPERT] for j in range(64)]


def rate(host, pool, size, order):
    n = pool // size
    g = torch.Generator().manual_seed(0)
    seq = torch.randint(0, n, (50000,), generator=g).tolist() if order == "random" else None
    best = 0.0
    for _ in range(3):
        i, nbytes = 0, 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            per = max(1, (64 * MiB) // size)
            for _ in range(per):
                s = (i % n) if seq is None else seq[i % len(seq)]
                if size == EXPERT:
                    dv[i % 64].copy_(host[s * size:(s + 1) * size], non_blocking=True)
                else:
                    dev[:size].copy_(host[s * size:(s + 1) * size], non_blocking=True)
                i += 1
            torch.cuda.synchronize()
            nbytes += per * size
        best = max(best, nbytes / (time.perf_counter() - t0) / 1e9)
    return round(best, 2)


for alloc in (torch_pin, thp_register):
    host, info = alloc()
    for pool_mib in (64, 320, 1024, 4096):
        for order in ("sequential", "random"):
            g = rate(host, pool_mib * MiB, EXPERT, order)
            print(json.dumps({"alloc": alloc.__name__, "pool_mib": pool_mib, "copy": "expert", "order": order,
                              "gbps": g, "us_per_expert": round(EXPERT / (g * 1e9) * 1e6, 1), **(info or {})}),
                  flush=True)
    g = rate(host, TOTAL, 64 * MiB, "sequential")
    print(json.dumps({"alloc": alloc.__name__, "pool_mib": 4096, "copy": "64MiB", "order": "sequential",
                      "gbps": g, **(info or {})}), flush=True)
    if info is not None:
        torch.cuda.cudart().cudaHostUnregister(host.data_ptr())
    del host

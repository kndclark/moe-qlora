"""H2D copy rate by copy size and source pool: separates per-copy overhead from DRAM/IOMMU effects.
One stream, back-to-back copies, synchronize every ~64 MiB; best of 3 x 1 s windows."""
import json, time, torch
MiB = 2**20
EXPERT = 5_611_520  # 5.35 MiB rounded up to 4 KiB; odd sizes misalign every copy
POOL = 2 * 2**30
host = torch.empty(POOL, dtype=torch.uint8, pin_memory=True)
host.fill_(1)
dev = torch.empty(1024 * MiB, dtype=torch.uint8, device="cuda")
for label, size in [("half_expert", EXPERT // 2), ("expert", EXPERT), ("4_experts", 4 * EXPERT),
                    ("64MiB", 64 * MiB), ("256MiB", 256 * MiB), ("1GiB", 1024 * MiB)]:
    for pool in ("reuse", "cycle_2GiB"):
        n = 1 if pool == "reuse" else POOL // size
        per_sync = max(1, (64 * MiB) // size)
        best = 0.0
        for _ in range(3):
            i, nbytes = 0, 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 1.0:
                for _ in range(per_sync):
                    o = (i % n) * size
                    d = (i % max(1, (1024 * MiB) // size)) * size
                    dev[d:d + size].copy_(host[o:o + size], non_blocking=True)
                    i += 1
                torch.cuda.synchronize()
                nbytes += per_sync * size
            best = max(best, nbytes / (time.perf_counter() - t0) / 1e9)
        print(json.dumps({"copy": label, "bytes": size, "pool": pool, "gbps": round(best, 2),
                          "us_per_expert": round(EXPERT / (best * 1e9) * 1e6, 1)}), flush=True)

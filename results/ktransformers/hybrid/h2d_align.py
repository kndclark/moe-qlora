"""Is the 36 GB/s expert-copy rate an alignment artefact? int(5.35 MiB) = 5,609,881 B is odd, so
every slice after the first starts on an odd address. Same byte count per copy, different strides /
offsets. 2 GiB torch-pinned pool, random order, sync every ~64 MiB, best of 3 x 1 s."""
import json, time, torch
MiB = 2**20
EXPERT = int(5.35 * MiB)
POOL = 2 * 2**30
host = torch.empty(POOL, dtype=torch.uint8, pin_memory=True)
host.fill_(1)
dev = torch.empty(POOL // 2, dtype=torch.uint8, device="cuda")


def rate(size, stride, shift):
    n = (POOL - shift - size) // stride
    nd = (dev.numel() - shift - size) // stride
    seq = torch.randint(0, n, (50000,), generator=torch.Generator().manual_seed(0)).tolist()
    best = 0.0
    for _ in range(3):
        i, nbytes = 0, 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            per = max(1, (64 * MiB) // size)
            for _ in range(per):
                s = shift + seq[i % len(seq)] * stride
                d = shift + (i % nd) * stride
                dev[d:d + size].copy_(host[s:s + size], non_blocking=True)
                i += 1
            torch.cuda.synchronize()
            nbytes += per * size
        best = max(best, nbytes / (time.perf_counter() - t0) / 1e9)
    return round(best, 2)


def up(x, a):
    return (x + a - 1) // a * a


arms = [
    ("expert, odd stride (my earlier probes)", EXPERT, EXPERT, 0),
    ("expert, 16 B stride", EXPERT, up(EXPERT, 16), 0),
    ("expert, 256 B stride", EXPERT, up(EXPERT, 256), 0),
    ("expert, 4 KiB stride", EXPERT, up(EXPERT, 4096), 0),
    ("expert rounded to 4 KiB, 4 KiB stride", up(EXPERT, 4096), up(EXPERT, 4096), 0),
    ("expert, 4 KiB stride, +1 B shift", EXPERT, up(EXPERT, 4096), 1),
    ("expert, 4 KiB stride, +8 B shift", EXPERT, up(EXPERT, 4096), 8),
    ("expert, 4 KiB stride, +64 B shift", EXPERT, up(EXPERT, 4096), 64),
    ("up row 2,494,464 B, own stride (Row L)", 2_494_464, 2_494_464, 0),
    ("up row 2,494,464 B, +1 B shift", 2_494_464, 2_494_464, 1),
]
for name, size, stride, shift in arms:
    g = rate(size, stride, shift)
    print(json.dumps({"arm": name, "bytes": size, "stride": stride, "shift": shift, "gbps": g,
                      "us_per_expert": round(EXPERT / (g * 1e9) * 1e6, 1)}), flush=True)

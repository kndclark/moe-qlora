"""Expert-sized H2D copies from a 4 GiB pinned pool (RAM, not cache), spread over 1/2/4/8 streams,
in sequential and random expert order (random = real misses). Best of 3 x 1 s windows."""
import json, time, torch
MiB = 2**20
EXPERT = 5_611_520  # 5.35 MiB rounded up to 4 KiB; odd sizes misalign every copy
N = (4 * 2**30) // EXPERT
host = torch.empty(N * EXPERT, dtype=torch.uint8, pin_memory=True)
host.fill_(1)
SLOTS = 64
dev = torch.empty(SLOTS * EXPERT, dtype=torch.uint8, device="cuda")
hv = [host[i * EXPERT:(i + 1) * EXPERT] for i in range(N)]
dv = [dev[j * EXPERT:(j + 1) * EXPERT] for j in range(SLOTS)]
g = torch.Generator().manual_seed(0)
rand_order = torch.randint(0, N, (100000,), generator=g).tolist()
for ns in (1, 2, 4, 8):
    streams = [torch.cuda.Stream() for _ in range(ns)]
    for order in ("sequential", "random"):
        best = 0.0
        for _ in range(3):
            i, nbytes = 0, 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 1.0:
                for k in range(12):  # 12 experts ~ two layer-steps of misses at batch 1
                    src = i % N if order == "sequential" else rand_order[i % len(rand_order)]
                    with torch.cuda.stream(streams[k % ns]):
                        dv[i % SLOTS].copy_(hv[src], non_blocking=True)
                    i += 1
                torch.cuda.synchronize()
                nbytes += 12 * EXPERT
            best = max(best, nbytes / (time.perf_counter() - t0) / 1e9)
        print(json.dumps({"streams": ns, "order": order, "gbps": round(best, 2),
                          "us_per_expert": round(EXPERT / (best * 1e9) * 1e6, 1)}), flush=True)

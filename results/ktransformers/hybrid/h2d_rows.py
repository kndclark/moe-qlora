"""Why per-row copies measured 48.5-50.9 GB/s in Row L (copy_bench.py) but 36-37 here.
Row L's method: 1 GiB pinned int32 random pool, k random rows of 2,494,464 B, the SAME k rows
copied 20 times, one sync at the end. Vary one thing at a time."""
import json, time, torch
ROW = 2_494_464
E = (1 << 30) // ROW
rand = torch.randint(-2**31, 2**31 - 1, (E, ROW // 4), dtype=torch.int32).pin_memory()
ones = torch.ones(E, ROW // 4, dtype=torch.int32).pin_memory()
dst = torch.empty(E, ROW // 4, dtype=torch.int32, device="cuda")
g = torch.Generator().manual_seed(0)


def run(pool, k, reps, fresh, sync_each):
    sets = [torch.randperm(E, generator=g)[:k].tolist() for _ in range(reps if fresh else 1)]
    def once():
        for r in range(reps):
            for e in sets[r % len(sets)]:
                dst[e].copy_(pool[e], non_blocking=True)
            if sync_each:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
    once()
    best = 0.0
    for _ in range(3):
        t0 = time.perf_counter()
        once()
        best = max(best, reps * k * ROW / (time.perf_counter() - t0) / 1e9)
    return round(best, 1)


for k in (6, 22, 68):
    for data, pool in (("random", rand), ("ones", ones)):
        for fresh in (False, True):
            for sync_each in (False, True):
                print(json.dumps({"rows": k, "data": data, "rows_per_rep": "fresh" if fresh else "same (Row L)",
                                  "sync": "per rep" if sync_each else "end (Row L)",
                                  "gbps": run(pool, k, 20, fresh, sync_each)}), flush=True)

"""Host->device copy load: back-to-back copies of one NVFP4 Lightning expert (5.35 MiB) from a
pinned pool larger than the CPU's last-level cache, reporting GB/s every interval with wall time."""
import argparse, json, time, torch
ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=600)
ap.add_argument("--pool-gib", type=float, default=2.0)
ap.add_argument("--interval", type=float, default=0.25)
ap.add_argument("--out", required=True)
a = ap.parse_args()
CH = 5_611_520  # 5.35 MiB rounded up to 4 KiB; odd sizes misalign every copy
n = int(a.pool_gib * 2**30) // CH
host = torch.empty(n * CH, dtype=torch.uint8, pin_memory=True)
host.fill_(1)
SLOTS = 8
dev = torch.empty(SLOTS * CH, dtype=torch.uint8, device="cuda")
hv = [host[i * CH:(i + 1) * CH] for i in range(n)]
dv = [dev[j * CH:(j + 1) * CH] for j in range(SLOTS)]
f = open(a.out, "w")
def emit(**kw):
    f.write(json.dumps(kw) + "\n"); f.flush()
emit(ready=time.time(), pool_experts=n, expert_bytes=CH)
end = time.perf_counter() + a.seconds
i, nbytes, t0 = 0, 0, time.perf_counter()
while time.perf_counter() < end:
    for j in range(SLOTS):
        dv[j].copy_(hv[i % n], non_blocking=True)
        i += 1
    torch.cuda.synchronize()
    nbytes += SLOTS * CH
    t = time.perf_counter()
    if t - t0 >= a.interval:
        emit(t=time.time(), gbps=round(nbytes / (t - t0) / 1e9, 3))
        nbytes, t0 = 0, t

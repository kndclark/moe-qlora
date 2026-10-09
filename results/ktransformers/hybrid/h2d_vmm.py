"""Copy rates out of the host memory K6 actually uses (CUDA VMM host pages, MixedRows) vs torch
pinned and a THP buffer registered with cudaHostRegister. Per allocation: copy engine per expert
(4 KiB-aligned 5.35 MiB rows, random, sync every ~64 MiB), K6's Triton gather of k rows, and a
GPU kernel reading the whole range. vllm_kstage is mounted read-only at /k."""
import ctypes, json, mmap, sys, time, torch
sys.path.insert(0, "/k")
import vllm_kstage as ks

MiB = 2**20
ROW = 5_611_520
R = (2 * 2**30) // ROW
NB = R * ROW
dst = torch.empty(R, ROW, dtype=torch.uint8, device="cuda")
g = torch.Generator().manual_seed(0)


def dev_view(ptr):
    return torch.as_tensor(ks._Arr(ptr, NB), device="cuda").view(R, ROW)


def torch_pin():
    h = torch.empty(NB, dtype=torch.uint8, pin_memory=True)
    h.fill_(1)
    return h.view(R, ROW), dev_view(h.data_ptr()), None


def thp_register():
    m = mmap.mmap(-1, NB + 2 * MiB, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    base = ctypes.addressof(ctypes.c_char.from_buffer(m))
    off = (-base) % (2 * MiB)
    m.madvise(mmap.MADV_HUGEPAGE, off, NB)
    h = torch.frombuffer(m, dtype=torch.uint8, count=NB, offset=off)
    h.fill_(1)
    assert int(torch.cuda.cudart().cudaHostRegister(h.data_ptr(), NB, 0)) == 0
    return h.view(R, ROW), dev_view(h.data_ptr()), lambda: torch.cuda.cudart().cudaHostUnregister(h.data_ptr())


def vmm_host():
    mix = ks.MixedRows(R, ROW, 0)
    t = mix.tensor(torch.uint8)
    t.fill_(1)
    return None, t, mix.free


def ce(src):
    seq = torch.randint(0, R, (50000,), generator=g).tolist()
    best = 0.0
    for _ in range(3):
        i, nb = 0, 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            for _ in range(12):
                s = seq[i % len(seq)]
                dst[i % R].copy_(src[s], non_blocking=True)
                i += 1
            torch.cuda.synchronize()
            nb += 12 * ROW
        best = max(best, nb / (time.perf_counter() - t0) / 1e9)
    return round(best, 2)


def gather(view, k):
    orders = [torch.randperm(R, generator=g)[:k].to(torch.int32).cuda() for _ in range(20)]
    ks.gather(view, dst, orders[0], k); torch.cuda.synchronize()
    best = 0.0
    for _ in range(3):
        t0 = time.perf_counter()
        for o in orders:
            ks.gather(view, dst, o, k)
        torch.cuda.synchronize()
        best = max(best, 20 * k * ROW / (time.perf_counter() - t0) / 1e9)
    return round(best, 2)


for alloc in (torch_pin, thp_register, vmm_host):
    host, view, free = alloc()
    out = {"alloc": alloc.__name__}
    if host is not None:
        out["ce_h2d"] = ce(host)
    out["ce_from_device_view"] = ce(view)
    for k in (6, 22, 68):
        out[f"gather_k{k}"] = gather(view, k)
    out["kernel_read"] = round(ks.read_rate(view), 2)
    print(json.dumps(out), flush=True)
    del host, view
    torch.cuda.synchronize()
    if free:
        free()

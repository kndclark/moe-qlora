"""Why THP + cudaHostRegister reads faster: page size or the registration API? And can torch's own
pinned allocator get it from config alone (PYTORCH_CUDA_ALLOC_CONF=pinned_use_cuda_host_register:True
with GLIBC_TUNABLES=glibc.malloc.hugetlb=1)? argv: arm names. Per arm: pointer alignment, the
AnonHugePages it added, copy engine, Triton gather k6/k22/k68 (K6's miss copy), kernel read (UVA)."""
import ctypes, json, mmap, os, sys, time, torch
sys.path.insert(0, "/k")
import vllm_kstage as ks

MiB = 2**20
ROW = 5_611_520
R = (2 * 2**30) // ROW
NB = R * ROW
dst = torch.empty(R, ROW, dtype=torch.uint8, device="cuda")
g = torch.Generator().manual_seed(0)
rt = torch.cuda.cudart()


def anon_huge():
    for l in open("/proc/self/smaps_rollup"):
        if l.startswith("AnonHugePages:"):
            return int(l.split()[1]) * 1024
    return 0


def dev_view(ptr):
    return torch.as_tensor(ks._Arr(ptr, NB), device="cuda").view(R, ROW)


def reg(advice):
    m = mmap.mmap(-1, NB + 2 * MiB, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    base = ctypes.addressof(ctypes.c_char.from_buffer(m))
    off = (-base) % (2 * MiB)
    m.madvise(advice, off, NB)
    h = torch.frombuffer(m, dtype=torch.uint8, count=NB, offset=off)
    h.fill_(1)
    assert int(rt.cudaHostRegister(h.data_ptr(), NB, 0)) == 0
    return h, lambda: rt.cudaHostUnregister(h.data_ptr())


ARMS = {
    "reg_4k": lambda: reg(mmap.MADV_NOHUGEPAGE),
    "reg_thp": lambda: reg(mmap.MADV_HUGEPAGE),
    "torch_pin": lambda: (torch.empty(NB, dtype=torch.uint8, pin_memory=True).fill_(1), None),
}


def ce(src):
    seq = torch.randint(0, R, (50000,), generator=g).tolist()
    best = 0.0
    for _ in range(3):
        i, nb = 0, 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            for _ in range(12):
                dst[i % R].copy_(src[seq[i % len(seq)]], non_blocking=True)
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


for name in sys.argv[1:]:
    a0 = anon_huge()
    h, free = ARMS[name]()
    p = h.data_ptr()
    out = {"arm": name, "conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
           "glibc": os.environ.get("GLIBC_TUNABLES", ""), "ptr_mod_4k": p % 4096, "ptr_mod_2m": p % (2 * MiB),
           "anon_huge_gib": round((anon_huge() - a0) / 2**30, 2)}
    view = dev_view(p)
    out["ce_h2d"] = ce(h.view(R, ROW))
    for k in (6, 22, 68):
        out[f"gather_k{k}"] = gather(view, k)
    out["kernel_read"] = round(ks.read_rate(view), 2)
    print(json.dumps(out), flush=True)
    del h, view
    torch.cuda.synchronize()
    if free:
        free()

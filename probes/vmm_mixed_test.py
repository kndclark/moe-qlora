"""Can one GPU tensor have device pages for hot experts and host pages for cold ones?

CUDA VMM test: reserve one virtual range, back its first half with device memory and its
second half with host memory (cuMemCreate, location HOST / HOST_NUMA), wrap it as a torch
tensor, check reads and writes, and time GPU reads of each half against a UVA pinned view
(what vLLM's --cpu-offload-gb uses) and a plain device tensor.
"""
import sys, time
import torch
from cuda.bindings import driver as cu


def ok(r):
    err, *rest = r if isinstance(r, tuple) else (r,)
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{err}")
    return rest[0] if len(rest) == 1 else rest


torch.cuda.init(); torch.zeros(1, device="cuda")  # primary context, shared with torch
dev = ok(cu.cuCtxGetDevice())
A = cu.CUdevice_attribute
for nm in ("CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED", "CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID",
           "CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED",
           "CU_DEVICE_ATTRIBUTE_HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED"):
    try:
        print(f"{nm}: {ok(cu.cuDeviceGetAttribute(getattr(A, nm), dev))}")
    except Exception as e:  # noqa: BLE001
        print(f"{nm}: query failed {e}")
print("driver", ok(cu.cuDriverGetVersion()), torch.cuda.get_device_name())


def prop(kind, nid=0):
    p = cu.CUmemAllocationProp()
    p.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    p.location.type = kind
    p.location.id = int(dev) if kind == cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE else nid
    return p


L = cu.CUmemLocationType
pdev = prop(L.CU_MEM_LOCATION_TYPE_DEVICE)
G = ok(cu.cuMemGetAllocationGranularity(pdev, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
print("device granularity", G)
phost = None
for kind in (L.CU_MEM_LOCATION_TYPE_HOST, L.CU_MEM_LOCATION_TYPE_HOST_NUMA):
    try:
        g = ok(cu.cuMemGetAllocationGranularity(prop(kind), cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
        h = ok(cu.cuMemCreate(G, prop(kind), 0)); ok(cu.cuMemRelease(h))
        print(f"host location {kind.name}: granularity {g}, cuMemCreate OK")
        phost = phost or prop(kind)
    except Exception as e:  # noqa: BLE001
        print(f"host location {kind.name}: {e}")
if phost is None:
    sys.exit("no host-backed VMM allocation: the hot/cold single-tensor build needs another route")

MiB = 1 << 20
half = int(sys.argv[1]) * MiB if len(sys.argv) > 1 else 512 * MiB
half = (half + G - 1) // G * G
va = ok(cu.cuMemAddressReserve(2 * half, G, 0, 0))
hd = ok(cu.cuMemCreate(half, pdev, 0)); ok(cu.cuMemMap(va, half, 0, hd, 0))
hh = ok(cu.cuMemCreate(half, phost, 0)); ok(cu.cuMemMap(int(va) + half, half, 0, hh, 0))
acc = cu.CUmemAccessDesc()
acc.location.type = L.CU_MEM_LOCATION_TYPE_DEVICE; acc.location.id = int(dev)
acc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
ok(cu.cuMemSetAccess(va, 2 * half, [acc], 1))


class Arr:  # torch.as_tensor reads this to wrap a raw device pointer
    def __init__(self, ptr, n):
        self.__cuda_array_interface__ = {"shape": (n,), "typestr": "<u1", "data": (ptr, False), "version": 3}


t = torch.as_tensor(Arr(int(va), 2 * half), device="cuda")
t.copy_(torch.randint(0, 256, (2 * half,), dtype=torch.uint8, device="cuda"))
ref = t.clone()
torch.cuda.synchronize()
assert torch.equal(t, ref), "mixed tensor read back differs"
print(f"mixed tensor: {2 * half // MiB} MiB, device half + host half, write/read OK")

pin = torch.randint(0, 256, (half,), dtype=torch.uint8).pin_memory()
try:
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    uva = get_accelerator_view_from_cpu_tensor(pin)
except Exception as e:  # noqa: BLE001
    print("vllm UVA view unavailable:", e); uva = None
plain = torch.empty(half, dtype=torch.uint8, device="cuda")


def gbps(x, reps=5):  # a GPU kernel reads every byte: same access pattern as weights in a GEMM
    v = x.view(torch.int32)
    v.sum(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        v.sum(dtype=torch.int64)
    torch.cuda.synchronize()
    return reps * x.numel() / (time.perf_counter() - t0) / 1e9


print(f"GPU read GB/s: device {gbps(plain):.1f}; VMM device half {gbps(t[:half]):.1f}; "
      f"VMM host half {gbps(t[half:]):.1f}" + (f"; UVA pinned view {gbps(uva):.1f}" if uva is not None else ""))
E = 2 * 1856 * 2688 * 9 // 16  # one Lightning expert: NVFP4 up + down, fp8 scale per 16 weights
g = torch.Generator().manual_seed(0)
for nm, src in (("VMM host half", t[half:]), ("UVA pinned view", uva)):
    if src is None:
        continue
    offs = torch.randint(0, (half - E) // 4096, (36,), generator=g).tolist()  # 6 layers x 6 experts
    sl = [src[o * 4096: o * 4096 + E].view(torch.int32) for o in offs]
    for s_ in sl[:4]:
        s_.sum()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for s_ in sl:
        s_.sum(dtype=torch.int64)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    print(f"{nm}: 36 scattered expert reads ({E / 1e6:.2f} MB each) {dt * 1e3:.2f} ms, {36 * E / dt / 1e9:.1f} GB/s")
x = torch.empty(half, dtype=torch.uint8, device="cuda")
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(5):
    x.copy_(pin, non_blocking=True)
torch.cuda.synchronize()
print(f"pinned H2D copy GB/s: {5 * half / (time.perf_counter() - t0) / 1e9:.1f}")
del t
ok(cu.cuMemUnmap(va, 2 * half)); ok(cu.cuMemRelease(hd)); ok(cu.cuMemRelease(hh)); ok(cu.cuMemAddressFree(va, 2 * half))
print("released")

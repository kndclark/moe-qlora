"""KSTAGE_UVA_THP check: the patched get_accelerator_view_from_cpu_tensor returns a CUDA view
equal to the CPU tensor (uint8, bf16, fp8; one under 2 MiB keeps the stock pinned path), and a
kernel reading it (a sum) agrees with the stock view; then sum time, stock vs mirror, 1 GiB."""
import sys, time, torch
sys.path.insert(0, "/k")
import vllm_kstage as ks
from vllm.model_executor.model_loader import utils as mlu

stock = mlu.get_accelerator_view_from_cpu_tensor
ks._uva_thp(mlu)
mine = mlu.get_accelerator_view_from_cpu_tensor
g = torch.Generator().manual_seed(1)
for shape, dt in (((128, 2880, 1440), torch.uint8), ((4096, 3000), torch.bfloat16),
                  ((300, 1000), torch.uint8), ((2048, 2048), torch.float8_e4m3fn)):
    if dt == torch.bfloat16:
        t = torch.randn(shape, generator=g).to(dt)
    else:
        t = torch.randint(0, 120, shape, generator=g, dtype=torch.uint8).view(dt)
    v = mine(t)
    assert v.is_cuda and v.shape == t.shape and v.dtype == t.dtype
    assert torch.equal(v.cpu().view(torch.uint8), t.view(torch.uint8)), (shape, dt)
    s = stock(t.pin_memory())
    a, b = v.view(torch.uint8).to(torch.int64).sum(), s.view(torch.uint8).to(torch.int64).sum()
    assert int(a) == int(b) == int(t.view(torch.uint8).to(torch.int64).sum())
    print(f"ok {tuple(shape)} {dt} {t.numel() * t.element_size() / 2**20:.1f} MiB", flush=True)
nb = 1 << 30
t = torch.randint(0, 255, (nb // 4,), generator=g, dtype=torch.int32)
for name, v in (("stock pinned", stock(t.pin_memory())), ("THP mirror", mine(t))):
    for _ in range(2):
        v.sum()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        v.sum()
    torch.cuda.synchronize()
    print(f"{name}: kernel read (sum) {5 * nb / (time.perf_counter() - t0) / 1e9:.1f} GB/s", flush=True)
ks.log("uva test PASS")

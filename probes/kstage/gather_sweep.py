"""Can a GPU kernel read scattered expert rows out of pinned host memory as fast as the copy
engine? Sweeps the Triton gather's block size, warps and load hint, and compares a
copy-engine copy per row. Same mount as copy_bench.py; arg: rows to fetch (default 36)."""
import sys, time
import torch, triton, triton.language as tl
sys.path.insert(0, "/k")
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
import vllm_kstage as ks

ROW, E = 2_494_464, 128
k = int(sys.argv[1]) if len(sys.argv) > 1 else 36
cpu = torch.randint(-2**31, 2**31 - 1, (E, ROW // 4), dtype=torch.int32).pin_memory()
view = get_accelerator_view_from_cpu_tensor(cpu)
alias = ks.host_alias(view)
dst = torch.empty_like(view)
ids = torch.randperm(E, generator=torch.Generator().manual_seed(0))[:k].cuda().int()
order = ks.plan(ids, E)


@triton.jit
def g_plain(src, dst, order, row_words, BLOCK: tl.constexpr):
    e = tl.load(order + tl.program_id(0))
    o = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = (o < row_words) & (e >= 0)
    base = tl.maximum(e, 0).to(tl.int64) * row_words
    tl.store(dst + base + o, tl.load(src + base + o, mask=m), mask=m)


@triton.jit
def g_cg(src, dst, order, row_words, BLOCK: tl.constexpr):
    e = tl.load(order + tl.program_id(0))
    o = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = (o < row_words) & (e >= 0)
    base = tl.maximum(e, 0).to(tl.int64) * row_words
    tl.store(dst + base + o, tl.load(src + base + o, mask=m, cache_modifier=".cg"), mask=m,
             cache_modifier=".cs")


def rate(fn, reps=20):
    fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return reps * k * ROW / (time.perf_counter() - t0) / 1e9


rw = ROW // 4
print(torch.cuda.get_device_name(), f"{k} rows")
best = (0, None)
import os
BS = [int(x) for x in os.environ.get("SWEEP_B", "512,1024,2048,4096,8192,16384").split(",")]
WS = [int(x) for x in os.environ.get("SWEEP_W", "1,2,4,8,16").split(",")]
for name, kern in (("plain", g_plain), ("cg/cs", g_cg)):
    for B in BS:
        line = []
        for W in WS:
            r = rate(lambda: kern[(k, triton.cdiv(rw, B))](view, dst, order, rw, BLOCK=B, num_warps=W))
            line.append(f"w{W} {r:5.1f}")
            best = max(best, (r, f"{name} BLOCK={B} warps={W}"))
        print(f"  {name:5} BLOCK {B:5}: " + "  ".join(line))
ol = order[:k].tolist()
print(f"per-row DMA {rate(lambda: [dst[e].copy_(alias[e], non_blocking=True) for e in ol]):.1f} GB/s; "
      f"best kernel {best[0]:.1f} ({best[1]}); rows equal {torch.equal(dst[ids.long()].cpu(), cpu[ids.long().cpu()])}")

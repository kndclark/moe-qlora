"""Experts in RAM: which route moves expert rows from host memory to the card fastest?

Answers the questions the kstage plugin (vllm_kstage.py) depends on, on one GPU:
  1. does vLLM's UVA view keep its pinned host tensor alive (is it safe to drop ours)?
  2. is the view's device pointer the host pointer (so a CPU alias can be built from it)?
  3. GB/s for: a GPU kernel reading the view (what Marlin does now), a torch copy of the
     view (cudaMemcpy, copy engine or kernel?), a copy from a CPU alias (DMA), a Triton
     gather of scattered expert rows (K2), and the same reads from VMM host pages (K5)
  4. can the DMA copy and the gather be captured in a CUDA graph (decode runs in graphs)?

Run in the vLLM image:
  docker run --rm --gpus all --pull never -v $PWD/probes/kstage:/k:ro --entrypoint python3 \\
    vllm/vllm-openai:v0.29.0 /k/copy_bench.py [MiB]
"""
import ctypes, sys, time
import torch

sys.path.insert(0, "/k")
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
import vllm_kstage as ks

MiB = 1 << 20
ROW = 2_494_464  # one expert's NVFP4 up projection (Marlin layout), bytes
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
E = n * MiB // ROW
dev = torch.device("cuda")
print(torch.cuda.get_device_name(), f"{E} rows of {ROW} B = {E * ROW / 2**30:.2f} GiB")


def rate(fn, nbytes, reps=5):
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return reps * nbytes / (time.perf_counter() - t0) / 1e9


# 1 + 2: lifetime and pointer identity
cpu = torch.randint(-2**31, 2**31 - 1, (E, ROW // 4), dtype=torch.int32).pin_memory()
ref = cpu.clone()
view = get_accelerator_view_from_cpu_tensor(cpu)
same_ptr = view.data_ptr() == cpu.data_ptr()
hp = cpu.data_ptr()
del cpu
other = torch.zeros(E, ROW // 4, dtype=torch.int32).pin_memory()  # would reuse a freed block
print(f"view ptr == host ptr: {same_ptr}; freed block reused by next pin: {other.data_ptr() == hp}; "
      f"view intact after del: {torch.equal(view.cpu(), ref)}")
del other

alias = ks.host_alias(view)  # CPU tensor over the same pinned bytes
print(f"CPU alias: pinned={alias.is_pinned()}, equal={torch.equal(alias, ref)}")

dst = torch.empty_like(view)
plain = torch.empty_like(view)
nb = E * ROW
print(f"GB/s  device read {ks.read_rate(plain):.1f} | view read (Marlin's path) {ks.read_rate(view):.1f}"
      f" | copy view->dev {rate(lambda: dst.copy_(view), nb):.1f}"
      f" | copy alias->dev non_blocking {rate(lambda: dst.copy_(alias, non_blocking=True), nb):.1f}")

# 3: scattered rows, as decode touches them: 36 of the layer's rows (c=4 ~ 22, c=16 ~ 68)
g = torch.Generator().manual_seed(0)
for k in (6, 22, 36, 68):
    k = min(k, E)
    ids = torch.randperm(E, generator=g)[:k].to(dev, torch.int32)
    order_t = ks.plan(ids.view(1, -1), E)
    t_gather = rate(lambda: ks.gather(view, dst, order_t, k), k * ROW, reps=20)
    t_idx = rate(lambda: dst.index_copy_(0, ids.long(), view.index_select(0, ids.long())), k * ROW, reps=20)
    order = order_t[:k].long()
    t_dma = rate(lambda: [dst[e].copy_(alias[e], non_blocking=True) for e in order.tolist()], k * ROW, reps=20)
    ok_rows = torch.equal(dst[ids.long()].cpu(), ref[ids.long().cpu()])
    print(f"  {k:3} rows: triton gather {t_gather:.1f}, index_select {t_idx:.1f}, per-row DMA {t_dma:.1f} GB/s; rows equal {ok_rows}")

# 4: graph capture of plan + gather, and of a whole-tensor DMA
ids = torch.randperm(E, generator=g)[:36].to(dev, torch.int32).view(1, -1)
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(2):
        ks.gather(view, dst, ks.plan(ids, E))
torch.cuda.synchronize()
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    ks.gather(view, dst, ks.plan(ids, E))
dst.zero_(); ids.copy_(torch.randperm(E, generator=g)[:36].view(1, -1).to(dev, torch.int32))
gr.replay(); torch.cuda.synchronize()
il = ids.view(-1).long()
print(f"graph: plan+gather replayed with new ids, rows equal {torch.equal(dst[il].cpu(), ref[il.cpu()])}")
gd = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(gd):
        dst.copy_(alias, non_blocking=True)
    dst.zero_(); gd.replay(); torch.cuda.synchronize()
    print(f"graph: whole-tensor DMA captured, equal {torch.equal(dst.cpu(), ref)}")
except Exception as e:  # noqa: BLE001
    print(f"graph: whole-tensor DMA capture failed: {type(e).__name__}: {e}")

# K5: mixed device/host tensor, read rates and a row copy out of its host half
mix = ks.MixedRows(E, ROW, E // 2)
mt = mix.tensor(torch.int32)
mt.copy_(view)
print(f"VMM mixed {E} rows, {E // 2} device: equal {torch.equal(mt.cpu(), ref)}; GB/s device half "
      f"{ks.read_rate(mt[:E // 2]):.1f}, host half {ks.read_rate(mt[E // 2:]):.1f}, "
      f"copy host half->dev {rate(lambda: dst[E // 2:].copy_(mt[E // 2:]), (E - E // 2) * ROW):.1f}")
del mt; mix.free()
print("done")

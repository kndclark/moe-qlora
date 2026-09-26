"""Is the first-launch reservation specific to FlashInfer, or the driver's
local-memory (stack) reservation that any kernel with STACK>0 triggers?

--stack-limit N : cudaDeviceSetLimit(cudaLimitStackSize, N) before launching
--kernel fi_vertical | fi_horizontal | fi_simple | cupy_stack0 | cupy_stackN
Prints the per-stage cudaMemGetInfo free numbers.
"""
import argparse
import ctypes
import json
import sys

import torch

sys.path.insert(0, "/work")
import common  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stack-limit", type=int, default=None)
ap.add_argument("--kernel", default="fi_vertical")
ap.add_argument("--local-bytes", type=int, default=64)
args = ap.parse_args()
MiB = 1 << 20

torch.cuda.init()
torch.zeros(1, device="cuda")  # context + a first STACK:0 torch kernel
common.cudart_last_error()
rt = common._cudart
lim = ctypes.c_size_t()


def get_limit(which):
    rt.cudaDeviceGetLimit(ctypes.byref(lim), which)
    return lim.value


def free():
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info()[0] / MiB


rec = dict(kernel=args.kernel, stack_limit_req=args.stack_limit)
rec["stack_limit_default"] = get_limit(0)      # cudaLimitStackSize
rec["printf_fifo"] = get_limit(1)
rec["malloc_heap"] = get_limit(2)
prep = None
if args.kernel.startswith("fi_"):
    inp = common.make_inputs(16, None, "paged")
    from flashinfer.mamba.selective_state_update import get_selective_state_update_module
    get_selective_state_update_module(torch.device("cuda:0"), torch.float32, torch.bfloat16,
                                      torch.bfloat16, torch.float32, torch.int32, 64, 128,
                                      1, torch.int32, torch.int64)
    algo = args.kernel[3:]
    launch = lambda: common.call_flashinfer(inp, algo)  # noqa: E731
else:
    import cupy as cp
    n = args.local_bytes // 4
    src = r'''
extern "C" __global__ void k(float* out, const int* idx, int n) {
  float buf[%d];
  #pragma unroll 1
  for (int i = 0; i < %d; ++i) buf[i] = out[i] + threadIdx.x;
  // dynamic index forces buf into local memory (stack)
  out[threadIdx.x] = buf[idx[threadIdx.x] %% %d];
}''' % (max(n, 1), max(n, 1), max(n, 1))
    src0 = r'''
extern "C" __global__ void k(float* out, const int* idx, int n) {
  out[threadIdx.x] = out[threadIdx.x] + idx[threadIdx.x];
}'''
    mod = cp.RawModule(code=src0 if args.kernel == "cupy_stack0" else src,
                       options=("-arch=sm_120",))
    kern = mod.get_function("k")
    rec["cupy_local_size_bytes"] = kern.attributes["local_size_bytes"]
    rec["cupy_num_regs"] = kern.attributes["num_regs"]
    o = cp.zeros(1024, dtype=cp.float32)
    ix = cp.arange(1024, dtype=cp.int32)
    launch = lambda: kern((1,), (256,), (o, ix, cp.int32(1024)))  # noqa: E731

rec["free_before_setlimit"] = free()
if args.stack_limit is not None:
    rec["setlimit_rc"] = rt.cudaDeviceSetLimit(0, ctypes.c_size_t(args.stack_limit))
    rec["stack_limit_after_set"] = get_limit(0)
rec["free_before_launch"] = free()
launch()
torch.cuda.synchronize()
rec["err"] = common.cudart_last_error()[1]
rec["free_after_launch"] = free()
rec["drop_setlimit_mib"] = rec["free_before_setlimit"] - rec["free_before_launch"]
rec["drop_launch_mib"] = rec["free_before_launch"] - rec["free_after_launch"]
rec["stack_limit_after_launch"] = get_limit(0)
print("RESULT " + json.dumps(rec))

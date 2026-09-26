"""One process, one (algorithm, batch) first launch of the FlashInfer SSU.

Records driver-level free memory (cudaMemGetInfo) at each stage, optionally
pins a filler allocation so only --leave-free-mib MiB stay free before the
first launch, and decides whether the kernel ran by an out-sentinel (NaN
prefill) plus cudaGetLastError. Prints one JSON line.
"""
import argparse
import json
import sys

import torch

sys.path.insert(0, "/work")
import common  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--algo", default="auto")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--layout", default="paged")
ap.add_argument("--nslots", type=int, default=None)
ap.add_argument("--leave-free-mib", type=float, default=None)
args = ap.parse_args()

MiB = 1 << 20
torch.cuda.init()
dev = torch.device("cuda:0")
rec = dict(algo=args.algo, batch=args.batch, layout=args.layout)


def free():
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info()[0]


rec["free_after_ctx_mib"] = free() / MiB
from flashinfer.jit import env as _je  # noqa
rec["jit_dir"] = str(_je.FLASHINFER_JIT_DIR)
rec["patched"] = __import__("os").environ.get("SSU_PATCH_DEDUP") == "1"
inp = common.make_inputs(args.batch, args.nslots, args.layout)
# Load the JIT module (host .so + its fatbin registration) before measuring.
from flashinfer.mamba.selective_state_update import get_selective_state_update_module  # noqa
get_selective_state_update_module(dev, torch.float32, torch.bfloat16, torch.bfloat16,
                                  torch.float32, torch.int32, 64, 128, 1,
                                  torch.int32, torch.int64)
rec["free_after_inputs_and_so_mib"] = free() / MiB

filler = None
if args.leave_free_mib is not None:
    target = int(args.leave_free_mib * MiB)
    cur = free()
    n = cur - target
    if n > 0:
        filler = torch.empty(n, dtype=torch.uint8, device=dev)
    # Top up / trim with small chunks to land close to the target.
    rec["free_before_launch_mib"] = free() / MiB
else:
    rec["free_before_launch_mib"] = free() / MiB

err0 = common.cudart_last_error()
try:
    common.call_flashinfer(inp, args.algo)
    rec["py_exception"] = None
except Exception as e:  # noqa: BLE001
    rec["py_exception"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
try:
    torch.cuda.synchronize()
    rec["sync_exception"] = None
except Exception as e:  # noqa: BLE001
    rec["sync_exception"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
err1 = common.cudart_last_error()
rec["cudaGetLastError_after_call"] = err1[1]
rec["free_after_first_launch_mib"] = torch.cuda.mem_get_info()[0] / MiB
rec["first_launch_free_drop_mib"] = rec["free_before_launch_mib"] - rec["free_after_first_launch_mib"]
out = inp["out"]
rec["out_nan_frac"] = float(torch.isnan(out.float()).float().mean())
rec["kernel_ran"] = rec["out_nan_frac"] == 0.0 and err1[0] == 0 and rec["py_exception"] is None
# second launch: does the extra memory stay reserved / does it now succeed?
try:
    inp["out"].fill_(float("nan"))
    common.call_flashinfer(inp, args.algo)
    torch.cuda.synchronize()
    e2 = common.cudart_last_error()
    rec["second_launch_err"] = e2[1]
    rec["second_launch_nan_frac"] = float(torch.isnan(inp["out"].float()).float().mean())
except Exception as e:  # noqa: BLE001
    rec["second_launch_err"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
rec["free_end_mib"] = torch.cuda.mem_get_info()[0] / MiB
print("RESULT " + json.dumps(rec))

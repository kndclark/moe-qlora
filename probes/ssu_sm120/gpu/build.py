"""JIT-build (or load) the FlashInfer SSU module with the exact dtype/shape
specialisation vLLM uses on this GPU, and print where the .so lives."""
import sys
import torch
sys.path.insert(0, "/work")
from flashinfer.mamba.selective_state_update import get_selective_state_update_module
from flashinfer.jit.mamba.selective_state_update import get_selective_state_update_uri
from flashinfer.jit import env as jit_env
from flashinfer.compilation_context import CompilationContext

dev = torch.device("cuda:0")
print("capability", torch.cuda.get_device_capability(dev))
print("nvcc arch flags", CompilationContext().get_nvcc_flags_list(supported_major_versions=[10, 11, 12]))
args = (torch.float32, torch.bfloat16, torch.bfloat16, torch.float32, torch.int32,
        64, 128, 1, torch.int32, torch.int64)
mod = get_selective_state_update_module(dev, *args)
uri = get_selective_state_update_uri(torch.float32, torch.bfloat16, torch.bfloat16,
                                     torch.float32, torch.int32, None, 64, 128, 1,
                                     torch.int32, torch.int64) + "_sm100"
print("uri", uri)
print("jit_dir", jit_env.FLASHINFER_JIT_DIR)
print("so", jit_env.FLASHINFER_JIT_DIR / uri / (uri + ".so"))

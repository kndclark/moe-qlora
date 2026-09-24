"""H1: does NVIDIA's remote nemotron_h code make Lightning's experts bnb-quantizable?

Builds Lightning from its config.json on the meta device with NVIDIA's
modeling_nemotron_h.py (trust_remote_code), runs the same
replace_with_bnb_linear call as the original blocker measurement, and
counts parameters by storage class. Meta device: nothing is allocated,
no kernel runs, so a stub for mamba_ssm's rmsnorm_fn import is enough.

Run CPU-only (no --gpus): with CUDA visible, transformers' is_mamba_2_ssm_available()
reads the stub's version and raises InvalidVersion('N/A'). Measured 2026-09-24.
  docker run --rm --init -v CODE:/code:ro -v STUB:/stub:ro -v $PWD/probes:/probes:ro \
    -v $PWD/results:/out -e HF_HOME=/tmp/hf --user $(id -u):$(id -g) \
    --entrypoint python3 gpu-lab:training /probes/h1_remote_code_bnb_count.py LABEL
CODE = Super (rev 2dc98e2a) or Nano (rev bf77c317) configuration_/modeling_nemotron_h.py
plus Lightning's config.json with auto_map added (Nano's code also needs
layers_block_type rewritten as hybrid_override_pattern). STUB = mamba_ssm/ops/triton/
layernorm_gated.py defining rmsnorm_fn, with empty __init__.py files.

Mounts expected:
  /code  dir holding config.json (+auto_map), configuration_nemotron_h.py,
         modeling_nemotron_h.py
  /stub  dir holding a stub mamba_ssm package (import-only)
  /out   results dir
"""
import json
import sys
import time

sys.path.insert(0, "/stub")

import bitsandbytes as bnb
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations.bitsandbytes import replace_with_bnb_linear

label = sys.argv[1] if len(sys.argv) > 1 else "h1"

cfg = AutoConfig.from_pretrained("/code", trust_remote_code=True)
with torch.device("meta"):
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, trust_remote_code=True)

before = sum(p.numel() for p in m.parameters())
m = replace_with_bnb_linear(
    m,
    modules_to_not_convert=["lm_head"],
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4"),
)

q4 = other = expert_q4 = expert_other = 0
n_linear4bit = 0
fused3d = 0
for name, mod in m.named_modules():
    if isinstance(mod, bnb.nn.Linear4bit):
        n_linear4bit += 1
for name, p in m.named_parameters():
    n = p.numel()
    owner = m.get_submodule(name.rsplit(".", 1)[0])
    is_expert = ".experts." in name
    if p.ndim == 3:
        fused3d += n
    if isinstance(owner, bnb.nn.Linear4bit) and name.endswith(".weight"):
        q4 += n
        expert_q4 += n if is_expert else 0
    else:
        other += n
        expert_other += n if is_expert else 0

sample = [n for n, _ in m.named_modules() if ".experts.0." in n][:4]
res = {
    "label": label,
    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "bitsandbytes": bnb.__version__,
    "model_class": f"{type(m).__module__}.{type(m).__name__}",
    "params_total": before,
    "params_linear4bit": q4,
    "params_not_4bit": other,
    "expert_params_linear4bit": expert_q4,
    "expert_params_not_4bit": expert_other,
    "params_in_3d_tensors": fused3d,
    "n_linear4bit_modules": n_linear4bit,
    "sample_expert_modules": {n: type(m.get_submodule(n)).__name__ for n in sample},
    "not_4bit_breakdown": {},
}
for name, p in m.named_parameters():
    owner = m.get_submodule(name.rsplit(".", 1)[0])
    if not (isinstance(owner, bnb.nn.Linear4bit) and name.endswith(".weight")):
        key = type(owner).__name__ + "." + name.rsplit(".", 1)[1]
        res["not_4bit_breakdown"][key] = res["not_4bit_breakdown"].get(key, 0) + p.numel()
print(json.dumps(res, indent=1))
with open(f"/out/{label}.json", "w") as f:
    json.dump(res, f, indent=1)

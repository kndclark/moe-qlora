"""G7 smoke, part 1: build untrained placement-A LoRA adapters for Lightning.

Question (plan.md G7): does vLLM 0.29 apply LoRA to every module family G6
trains? Placement A = attention q/k/v/o, Mamba in_proj, shared-expert up/down
(TARGETS, as in g5_train_step.py). Serving these beside the base model answers
it by observation, per family:

  zero    all targets, lora_B = 0: must leave outputs identical to base (it loads)
  attn    q/k/v/o only, random          must change outputs (vLLM applies it)
  inproj  Mamba in_proj only, random    must change outputs
  shexp   shared experts only, random   must change outputs
  full    all targets, random           must change outputs

A family vLLM skips silently looks like "zero", which is why each gets its own
adapter. Weights are seeded; nothing is trained. The model is built on the meta
device, so no checkpoint is read and no GPU is used: PEFT supplies the key names
and shapes a real G6 adapter would have, and its own config file.

Run (CPU only):
  docker run --rm -v /srv/model-cache:/hf:ro -v ~/moe-qlora/probes:/probes:ro \
    -v OUTDIR:/out -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 --user $(id -u):$(id -g) \
    --entrypoint python3 gpu-lab:training /probes/g7_lora_adapters.py
"""
import json
import os
import re

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
REVISION = "a9904d24bcc1d289a1950fa9d2b978c47cf903b9"
TARGETS = r".*\.mixer\.(q_proj|k_proj|v_proj|o_proj|in_proj)$|.*\.mixer\.shared_experts\.(up_proj|down_proj)$"
FAMILIES = {
    "attn": r"\.mixer\.(q_proj|k_proj|v_proj|o_proj)\.",
    "inproj": r"\.mixer\.in_proj\.",
    "shexp": r"\.mixer\.shared_experts\.",
}
SEED = 20260927

cfg = AutoConfig.from_pretrained(MODEL, revision=REVISION)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg)
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                  target_modules=TARGETS)
pm = get_peft_model(model, lcfg)
pm.peft_config["default"].base_model_name_or_path = MODEL
shapes = {k: tuple(v.shape) for k, v in get_peft_model_state_dict(pm).items()}

# Every key belongs to exactly one family, or the per-family verdicts overlap.
for k in shapes:
    hits = [f for f, rx in FAMILIES.items() if re.search(rx, k)]
    assert len(hits) == 1, (k, hits)

gen = torch.Generator().manual_seed(SEED)


def build(name, keep, zero_b):
    tensors = {}
    for k, s in sorted(shapes.items()):
        if not re.search(keep, k):
            continue
        if ".lora_A." in k:
            t = torch.randn(s, generator=gen) / s[1] ** 0.5
        elif zero_b:
            t = torch.zeros(s)
        else:
            t = torch.randn(s, generator=gen) * 0.02
        tensors[k] = t.to(torch.bfloat16).contiguous()
    d = f"/out/{name}"
    os.makedirs(d, exist_ok=True)
    save_file(tensors, f"{d}/adapter_model.safetensors")
    pm.peft_config["default"].save_pretrained(d)
    return {"modules": len(tensors) // 2,
            "params": sum(t.numel() for t in tensors.values()),
            "example_key": next(iter(tensors))}


report = {"model": MODEL, "revision": REVISION, "targets": TARGETS, "r": 16, "alpha": 32,
          "seed": SEED, "keys_total": len(shapes)}
report["zero"] = build("zero", r".", zero_b=True)
for fam, rx in FAMILIES.items():
    report[fam] = build(fam, rx, zero_b=False)
report["full"] = build("full", r".", zero_b=False)
print(json.dumps(report, indent=1))
with open("/out/adapters.json", "w") as f:
    json.dump(report, f, indent=1)

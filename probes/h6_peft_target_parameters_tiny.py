"""H6: does stock PEFT 0.21 target_parameters accept bnb-parametrized 4-bit experts?

Same tiny nemotron_h as h3. Experts quantized with
bnb.nn.parametrize.replace_parameter_4bit, then LoraConfig(target_parameters=
["experts.up_proj", "experts.down_proj"]) with no axolotl patch. Records
whether injection succeeds, whether expert LoRA params get gradients,
whether the loss moves, and whether the experts stay uint8.

Mounts: /out results dir. Env: HF_HUB_OFFLINE=1.
"""
import json
import sys
import time
import traceback

import bitsandbytes as bnb
import peft
import torch
import transformers
from bitsandbytes.nn.parametrize import replace_parameter_4bit
from transformers import NemotronHConfig, NemotronHForCausalLM
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts

label = sys.argv[1] if len(sys.argv) > 1 else "h6"
quantize = "noquant" not in sys.argv[2:]  # control: same targets on bf16 experts
torch.manual_seed(0)
cfg = NemotronHConfig(
    vocab_size=1024, hidden_size=256,
    layers_block_type=["mamba", "moe", "attention", "moe"],
    num_attention_heads=4, num_key_value_heads=2, head_dim=64, intermediate_size=512,
    mamba_num_heads=8, mamba_head_dim=32, ssm_state_size=16, n_groups=2, chunk_size=32,
    n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=128,
    moe_shared_expert_intermediate_size=256, n_group=1, topk_group=1,
    routed_scaling_factor=2.5, mlp_hidden_act="relu2", num_nextn_predict_layers=0,
    use_cache=False,
)
m = NemotronHForCausalLM(cfg).to("cuda", torch.bfloat16)
for mod in m.modules():
    if quantize and isinstance(mod, NemotronHExperts):
        for pname in ("up_proj", "down_proj"):
            replace_parameter_4bit(mod, pname, compress_statistics=True, quant_type="nf4")
for p in m.parameters():
    p.requires_grad_(False)

res = {"label": label, "quantized_experts": quantize, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "torch": torch.__version__,
       "transformers": transformers.__version__, "bitsandbytes": bnb.__version__, "peft": peft.__version__}
ids = torch.randint(0, cfg.vocab_size, (2, 96), device="cuda")
try:
    lcfg = peft.LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=[],
                           target_parameters=["experts.up_proj", "experts.down_proj"])
    pm = peft.get_peft_model(m, lcfg)
    names = [n for n, p in pm.named_parameters() if p.requires_grad]
    res["inject"] = "ok"
    res["trainable_names_sample"] = names[:4]
    res["trainable_params"] = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-3)
    losses = []
    for i in range(6):
        loss = pm(input_ids=ids, labels=ids).loss
        loss.backward()
        if i == 0:
            res["expert_lora_with_grad"] = sum(1 for n, p in pm.named_parameters()
                                               if p.requires_grad and p.grad is not None
                                               and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(round(loss.item(), 5))
    res["losses"] = losses
    res["experts_still_uint8"] = quantize and all(
        mod.parametrizations["up_proj"].original.dtype == torch.uint8
        for mod in pm.modules() if isinstance(mod, NemotronHExperts))
except Exception as e:  # the refusal itself is the result
    res["inject_or_train"] = "error"
    res["error"] = f"{type(e).__name__}: {e}"[:600]
    res["traceback_tail"] = traceback.format_exc().splitlines()[-6:]
print(json.dumps(res, indent=1))
with open(f"/out/{label}.json", "w") as f:
    json.dump(res, f, indent=1)

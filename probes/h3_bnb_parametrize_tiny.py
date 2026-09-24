"""H3: does bnb 0.50.2's replace_parameter_4bit work on transformers' fused NemotronHExperts?

Tiny random nemotron_h (built-in transformers class, fused 3D experts) on one
GPU. Three copies of the same weights:
  ref  bf16 experts as initialised
  q    experts replaced by bnb.nn.parametrize.replace_parameter_4bit (NF4)
  deq  bf16 experts set to dequantize(q's NF4), i.e. q's weights held densely
q vs deq isolates the mechanism (should agree to bf16 noise); q vs ref is
the NF4 quantisation error. Backward: gradients must flow THROUGH the
quantised MoE layers to reach layer 0, so layer 0's weights are the probe.
Then standard PEFT LoRA on the non-expert Linears and a few AdamW steps
with gradient checkpointing, to see the loss move.

Mounts: /out results dir. Env: HF_HUB_OFFLINE=1 (keeps the kernels hub
from downloading; mamba falls back to the torch path).
"""
import copy
import json
import sys
import time

import bitsandbytes as bnb
import peft
import torch
import transformers
from bitsandbytes.nn.parametrize import replace_parameter_4bit
from transformers import NemotronHConfig, NemotronHForCausalLM
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts

label = sys.argv[1] if len(sys.argv) > 1 else "h3"
dev = "cuda"
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
ref = NemotronHForCausalLM(cfg).to(dev, torch.bfloat16).eval()
experts_impl = getattr(ref.config, "_experts_implementation", None)


def experts_modules(m):
    return [(n, mod) for n, mod in m.named_modules() if isinstance(mod, NemotronHExperts)]


q = copy.deepcopy(ref)
bytes_before = bytes_after = 0
for _, mod in experts_modules(q):
    for pname in ("up_proj", "down_proj"):
        p = getattr(mod, pname)
        bytes_before += p.numel() * p.element_size()
        replace_parameter_4bit(mod, pname, compress_statistics=True, quant_type="nf4")
        orig = mod.parametrizations[pname].original
        bytes_after += orig.numel() * orig.element_size()
q_types = sorted({str(mod.parametrizations["up_proj"].original.dtype) for _, mod in experts_modules(q)})

deq = copy.deepcopy(ref)
with torch.no_grad():
    for (_, mq), (_, md) in zip(experts_modules(q), experts_modules(deq)):
        for pname in ("up_proj", "down_proj"):
            getattr(md, pname).copy_(getattr(mq, pname))  # parametrized access = dequantize

ids = torch.randint(0, cfg.vocab_size, (2, 96), device=dev)
with torch.no_grad():
    lr, lq, ld = (m(input_ids=ids).logits.float() for m in (ref, q, deq))


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


fwd = {"q_vs_deq_rel": rel(lq, ld), "q_vs_ref_rel": rel(lq, lr),
       "q_vs_deq_maxabs": (lq - ld).abs().max().item(), "finite": bool(torch.isfinite(lq).all())}

# Backward through the quantised MoE layers, compared with the dense twin.
probe = "model.layers.0.mixer.in_proj.weight"
grads = {}
for tag, m in (("q", q), ("deq", deq)):
    m.train()
    for n, p in m.named_parameters():
        p.requires_grad_(n == probe)
    out = m(input_ids=ids, labels=ids)
    out.loss.backward()
    g = dict(m.named_parameters())[probe].grad
    grads[tag] = g.float().clone()
    n_expert_grads = sum(1 for n, p in m.named_parameters() if ".experts." in n and p.grad is not None)
    grads[tag + "_expert_params_with_grad"] = n_expert_grads
bwd = {"probe": probe, "grad_q_vs_deq_rel": rel(grads["q"], grads["deq"]),
       "grad_norm_q": grads["q"].norm().item(), "grad_finite": bool(torch.isfinite(grads["q"]).all()),
       "expert_params_with_grad_q": grads["q_expert_params_with_grad"]}

# PEFT LoRA on non-expert Linears + gradient checkpointing, a few AdamW steps.
for p in q.parameters():
    p.requires_grad_(False)
    p.grad = None
q.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
lcfg = peft.LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                       # PEFT 0.21 refuses mamba out_proj/conv1d on nemotron_h (measured).
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj",
                                       "shared_experts.up_proj", "shared_experts.down_proj"])
pm = peft.get_peft_model(q, lcfg)
trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad)
opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-3)
torch.cuda.reset_peak_memory_stats()
losses = []
for _ in range(8):
    loss = pm(input_ids=ids, labels=ids).loss
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    losses.append(round(loss.item(), 5))
still_4bit = all(mod.parametrizations["up_proj"].original.dtype == torch.uint8
                 for _, mod in experts_modules(pm))

res = {
    "label": label, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
    "transformers": transformers.__version__, "bitsandbytes": bnb.__version__, "peft": peft.__version__,
    "experts_implementation": experts_impl,
    "n_expert_modules": len(experts_modules(q)),
    "expert_bytes_bf16": bytes_before, "expert_bytes_packed": bytes_after, "packed_dtypes": q_types,
    "forward": fwd, "backward": bwd,
    "lora": {"trainable_params": trainable, "losses": losses, "experts_still_uint8": still_4bit,
             "peak_alloc_MiB": round(torch.cuda.max_memory_allocated() / 2**20, 1)},
}
print(json.dumps(res, indent=1))
with open(f"/out/{label}.json", "w") as f:
    json.dump(res, f, indent=1)

"""F3 pre-run checks (plan.md): expert_lora.py and lm_head LoRA under chunked_ce.py, against
references, before any real-model run. CPU fp32 on a tiny nemotron_h; then a GPU smoke of the
per-expert grouped matmuls at the real shapes for r = 4, 8, 16.

  (a) B = 0: the LoRA experts forward is bitwise the stock grouped_mm_experts_forward.
  (b) random A, B: output and factor gradients against a dense reference (stock forward on
      W + s * (A B)^T, gradients through that delta).
  (c) whole tiny model, gradient checkpointing on, placement-A PEFT LoRA + lm_head LoRA +
      expert LoRA: chunked loss vs the stock loss, and every trainable gradient.

  ... --entrypoint python3 gpu-lab:training /probes/f3_check.py
"""
import json
import sys
import types
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import NemotronHConfig, NemotronHForCausalLM
from transformers.integrations.moe import grouped_mm_experts_forward
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chunked_ce  # noqa: E402
import expert_lora  # noqa: E402

TARGETS = r".*\.mixer\.(q_proj|k_proj|v_proj|o_proj|in_proj)$|.*\.mixer\.shared_experts\.(up_proj|down_proj)$"
res = {"torch": torch.__version__}


def tiny():
    torch.manual_seed(0)
    cfg = NemotronHConfig(
        vocab_size=1024, hidden_size=256, layers_block_type=["mamba", "moe", "attention", "moe"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, intermediate_size=512,
        mamba_num_heads=8, mamba_head_dim=32, ssm_state_size=16, n_groups=2, chunk_size=32,
        n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=256, n_group=1, topk_group=1, routed_scaling_factor=2.5,
        mlp_hidden_act="relu2", num_nextn_predict_layers=0, use_cache=False, tie_word_embeddings=False)
    m = NemotronHForCausalLM(cfg).float()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def rel(a, b):
    return float((a.float() - b.float()).abs().max() / b.float().abs().max())


# ---- (a), (b): one experts module, both modes
for mode in ("shared", "per_expert"):
    m = tiny()
    e = next(mod for mod in m.modules() if isinstance(mod, NemotronHExperts))
    torch.manual_seed(1)
    x = torch.randn(64, 256)
    idx = torch.stack([torch.randperm(8)[:2] for _ in range(64)])
    w = torch.rand(64, 2)
    with torch.no_grad():
        stock = grouped_mm_experts_forward(e, x, idx, w)
    params = expert_lora.attach_expert_lora(m, r=8, alpha=16, mode=mode, dtype=torch.float32)
    r = {"n_modules": len(params) // 4, "shapes": [list(p.shape) for p in params[:4]]}
    with torch.no_grad():
        r["b0_bitwise_equal_to_stock"] = bool(torch.equal(e(x, idx, w), stock))
        for n in ("lora_up_B", "lora_down_B"):
            getattr(e, n).normal_(0, 0.05)
    go = torch.randn(64, 256)
    y = e(x, idx, w)
    y.backward(go)
    gl = {n: getattr(e, n).grad.clone() for n in expert_lora.LORA_NAMES}
    # dense reference: stock forward on W + delta, delta built from the same factors
    fac = {n: getattr(e, n).detach().clone().requires_grad_(True) for n in expert_lora.LORA_NAMES}
    Wu, Wd = e.up_proj.detach(), e.down_proj.detach()
    f = [fac[n] if mode == "per_expert" else fac[n].expand(8, *fac[n].shape) for n in expert_lora.LORA_NAMES]
    s = e.lora_scaling

    d = types.SimpleNamespace(num_experts=8, has_gate=False, has_bias=False, is_transposed=False, act_fn=e.act_fn,
                              up_proj=Wu + s * torch.bmm(f[0], f[1]).transpose(1, 2),
                              down_proj=Wd + s * torch.bmm(f[2], f[3]).transpose(1, 2))
    y_ref = grouped_mm_experts_forward(d, x, idx, w)
    y_ref.backward(go)
    r["out_rel_max_diff_vs_dense"] = rel(y.detach(), y_ref.detach())
    r["grad_rel_max_diff_vs_dense"] = {n: rel(gl[n], fac[n].grad) for n in expert_lora.LORA_NAMES}
    r["out_rel_change_vs_stock"] = rel(y.detach() - stock, stock)
    res[f"module_{mode}"] = r
    expert_lora._track.clear()

# ---- (c): whole model, checkpointing, placement A + lm_head + experts, chunked vs stock loss
for mode in ("shared", "per_expert"):
    m = tiny()
    m.config.use_cache = False
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.enable_input_require_grads()
    ep = expert_lora.attach_expert_lora(m, r=8, alpha=16, mode=mode, dtype=torch.float32)
    pm = get_peft_model(m, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                      target_modules=TARGETS + "|lm_head"))
    for p in ep:
        p.requires_grad_(True)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if p.requires_grad and ("lora_B" in n or n.endswith("_B")):
                p.normal_(0, 0.02)
    names = [n for n, p in pm.named_parameters() if p.requires_grad]
    kinds = sorted({n.split(".")[-2] if "lora_" in n.split(".")[-2] else n.split(".")[-1] for n in names})
    ids = torch.randint(0, 1024, (1, 96))
    out = {}
    for how in ("stock", "chunked"):
        (chunked_ce.install(32) if how == "chunked" else chunked_ce.uninstall())
        pm.zero_grad(set_to_none=True)
        loss = pm(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
        out[how] = (loss.item(), {n: p.grad.clone() for n, p in pm.named_parameters() if p.requires_grad})
    chunked_ce.uninstall()
    (ls, gs), (lc, gc) = out["stock"], out["chunked"]
    lm = [n for n in names if "lm_head" in n]
    res[f"model_{mode}"] = {
        "trainable_tensors": len(names), "lm_head_lora_tensors": lm, "expert_lora_tensors": len(ep),
        "all_grads_present": all(g is not None for g in gc.values()),
        "all_grads_nonzero": all(float(g.abs().sum()) > 0 for g in gc.values()),
        "loss_stock_vs_chunked": [ls, lc, abs(ls - lc)],
        "grad_rel_max_diff_worst": max(rel(gc[n], gs[n]) for n in names),
        "grad_rel_max_diff_lm_head": {n.split("lm_head.")[1]: rel(gc[n], gs[n]) for n in lm},
        "chunked_calls": chunked_ce.state["calls"], "expert_forward_calls": expert_lora.state["calls"]}
    expert_lora._track.clear()

# ---- GPU smoke: per-expert grouped matmuls at the real shapes (E 128, H 2688, I 1856), bf16
if torch.cuda.is_available():
    from transformers.integrations.moe import _grouped_mm
    S = 6 * 1024
    offs = torch.cumsum(torch.full((128,), S // 128, dtype=torch.int32, device="cuda"), 0, dtype=torch.int32)
    x = torch.randn(S, 2688, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    for rank in (4, 8, 16):
        try:
            A = (torch.randn(128, 2688, rank, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_(True)
            B = (torch.randn(128, rank, 1856, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_(True)
            y = _grouped_mm(_grouped_mm(x, A, offs), B, offs)
            y.float().square().mean().backward()
            ref = torch.cat([x[i * 48:(i + 1) * 48].float() @ A[i].float() @ B[i].float() for i in range(128)])
            res[f"gpu_per_expert_r{rank}"] = {"ok": True, "rel_max_diff_vs_fp32_loop": rel(y.detach(), ref),
                                              "grads_finite": bool(A.grad.isfinite().all() and B.grad.isfinite().all())}
        except Exception as e:
            res[f"gpu_per_expert_r{rank}"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        x.grad = None
print(json.dumps(res, indent=1))

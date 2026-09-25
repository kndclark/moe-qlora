"""LoRA on the routed experts of transformers 5.16.1 NemotronHExperts (F3 in plan.md).

Why: stock PEFT 0.21 cannot do it here. target_parameters needs full paths, then crashes on
the first checkpointed backward (_remove_parametrizations ignores _LoraFactorsProxy), and
axolotl v0.19.0's patch fixes the matching but not the crash; patched further it trains but
builds the full (128, out, in) weight delta every forward, +1.19 GiB per layer (MEASURED on
a CPU tiny model by a research agent, hand-off #4). This adds the LoRA to the activations
inside the experts forward instead, so no (E, out, in) delta is ever built and the 4-bit
expert weights stay as they are.

Drafted by a research agent 2026-09-25 (per-expert mode; CPU tiny model: 4.6e-7 relative
to a dense fp32 reference, identical with and without checkpointing); the shared mode is
added here. Two layouts:
  per_expert: A_up (E, H, r), B_up (E, r, I), A_down (E, I, r), B_down (E, r, H)
  shared:     A_up (H, r), B_up (r, I), A_down (I, r), B_down (r, H), one pair per
              projection per layer applied to every expert; this is Megatron-Bridge's
              default (share_expert_adapters=True, peft/lora.py:156 at 03e4dcc5)
    up   = x @ W_up[e]^T + s * (x @ A_up[e]) @ B_up[e];   h = relu2(up)
    down = h @ W_down[e]^T + s * (h @ A_down[e]) @ B_down[e]
In shared mode the up term is the same for every expert a token visits, so it is computed
once per token and gathered. Factors may be stored in fp32; the LoRA terms are computed in
the activation dtype (bf16), as the base grouped matmuls are.

The forward is transformers 5.16.1 grouped_mm_experts_forward (integrations/moe.py:377)
specialised to has_gate=False, has_bias=False, is_transposed=False, without the
expert-parallel sentinel masks (no expert parallelism here, so they are no-ops), plus the
two LoRA terms. With B = 0 it is the stock path.

Order matters: PEFT's get_peft_model freezes every parameter outside its own tuner layers,
so call p.requires_grad_(True) on the returned factors after it. PEFT will not save these;
use expert_lora_state_dict().
"""
import math
import types

import torch
from torch import nn
from transformers.integrations.moe import _grouped_linear, _grouped_mm
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts

LORA_NAMES = ("lora_up_A", "lora_up_B", "lora_down_A", "lora_down_B")
state = {"calls": 0, "modules": 0, "mode": None, "r": None, "alpha": None, "dtype": None, "params": 0}
_track = []  # (name, param, sample positions, their values at attach time) for the "did it train" check


def grouped_mm_lora_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)
    expert_ids_g, perm = torch.sort(expert_ids)
    rows = perm // num_top_k
    x = hidden_states[rows]  # (S, H)
    sample_weights_g = sample_weights[perm]
    histc_input = expert_ids_g.float() if device.type in ("cpu", "mps") else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    s, dt = self.lora_scaling, x.dtype
    Au, Bu, Ad, Bd = (getattr(self, n).to(dt) for n in LORA_NAMES)
    up = _grouped_linear(x, self.up_proj, offsets)  # (S, I)
    if self.lora_mode == "shared":
        up = up + ((hidden_states @ Au) @ Bu * s)[rows]
    else:
        up = up + _grouped_mm(_grouped_mm(x, Au, offsets), Bu, offsets) * s
    h = self.act_fn(up)  # relu2
    out = _grouped_linear(h, self.down_proj, offsets)  # (S, H)
    if self.lora_mode == "shared":
        out = out + (h @ Ad) @ Bd * s
    else:
        out = out + _grouped_mm(_grouped_mm(h, Ad, offsets), Bd, offsets) * s
    state["calls"] += 1

    weighted_out = out * sample_weights_g.unsqueeze(-1)  # fp32 (router weights are fp32), as in the stock path
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)
    weighted_out = weighted_out[inv_perm]
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
    return final_hidden_states.to(hidden_states.dtype)


def _param_device(mod, name):
    if hasattr(mod, "parametrizations") and name in mod.parametrizations:
        return mod.parametrizations[name].original.device  # do not touch mod.<name>: that dequantises
    return getattr(mod, name).device


def attach_expert_lora(model, r=16, alpha=32, mode="per_expert", dtype=torch.bfloat16):
    """Add LoRA factors to every NemotronHExperts and swap in the LoRA forward.
    A ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)) (= PEFT's kaiming_uniform_(a=sqrt(5)) default), B = 0.
    Returns the new Parameters (requires_grad=True)."""
    assert mode in ("per_expert", "shared"), mode
    params = []
    for name, mod in model.named_modules():
        if not isinstance(mod, NemotronHExperts):
            continue
        assert not mod.has_gate and not mod.has_bias and not mod.is_transposed, "forward assumes NemotronH layout"
        E, I = mod.num_experts, mod.intermediate_dim
        H = mod.config.moe_latent_size or mod.hidden_dim
        dev = _param_device(mod, "up_proj")
        lead = (E,) if mode == "per_expert" else ()

        def a(fan_in, fan_out):
            b = 1.0 / math.sqrt(fan_in)
            return nn.Parameter(torch.empty(*lead, fan_in, fan_out, device=dev, dtype=dtype).uniform_(-b, b))

        mod.lora_up_A = a(H, r)
        mod.lora_up_B = nn.Parameter(torch.zeros(*lead, r, I, device=dev, dtype=dtype))
        mod.lora_down_A = a(I, r)
        mod.lora_down_B = nn.Parameter(torch.zeros(*lead, r, H, device=dev, dtype=dtype))
        mod.lora_scaling = alpha / r
        mod.lora_mode = mode
        mod.forward = types.MethodType(grouped_mm_lora_experts_forward, mod)  # instance attr beats class forward
        for n in LORA_NAMES:
            p = getattr(mod, n)
            params.append(p)
            pos = torch.linspace(0, p.numel() - 1, 256, device=dev).long()  # spread over every expert
            _track.append((f"{name}.{n}", p, pos, p.detach().flatten()[pos].clone()))
        state["modules"] += 1
    state.update(mode=mode, r=r, alpha=alpha, dtype=str(dtype), params=sum(p.numel() for p in params))
    return params


def expert_lora_state_dict(model):
    return {n: p.detach().contiguous() for n, p in model.named_parameters() if n.rsplit(".", 1)[-1] in LORA_NAMES}


def dense_delta(mod):
    """(delta_up, delta_down) in the base layout (E, out, in), fp32 -- for merging or checks only."""
    s = mod.lora_scaling
    f = [getattr(mod, n).float() for n in LORA_NAMES]
    if mod.lora_mode == "shared":
        f = [t.expand(mod.num_experts, *t.shape) for t in f]
    return s * torch.bmm(f[0], f[1]).transpose(1, 2), s * torch.bmm(f[2], f[3]).transpose(1, 2)


def report():
    """How many factors moved from their initial values: B starts at 0, so a moved B means the
    optimizer stepped it; a moved A means gradient reached A too (it only can once B != 0)."""
    moved = {n: 0 for n in LORA_NAMES}
    for name, p, pos, init in _track:
        if not torch.equal(p.detach().flatten()[pos], init):
            moved[name.rsplit(".", 1)[-1]] += 1
    return dict(state, factors_moved_of=state["modules"], factors_moved=moved)

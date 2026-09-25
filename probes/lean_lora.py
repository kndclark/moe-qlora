"""A leaner LoRA backward for PEFT 0.21's Linear4bit and Linear layers (F2 in plan.md).

Why (David asked for it 2026-09-25, from the NVIDIA research): PEFT's default
autocast_adapter_dtype=True keeps the LoRA factors in fp32, so every LoRA branch casts its
bf16 input to fp32 and lora_A's nn.Linear saves that fp32 copy for its backward
(tokens x in_features x 4 bytes, 21 MiB at 2048 x 2688), and lora_B saves the fp32
A-activation (tokens x r x 4). Here the branch is one autograd Function that saves only the
bf16 input (usually already alive, so often free) and the two factors (parameters, free),
and recomputes x.float() @ A^T in the backward. Under gradient checkpointing this only
matters inside the layer being recomputed, so the expected gain is tens of MiB
(ARITHMETIC, hand-off #4), not GiB.

The idea is NVIDIA NeMo AutoModel's LoRATritonFunction (components/_peft/lora.py:702-818 at
d9334f362fdb3cc71358155961c7b3b402259d75): save x, A, B; recompute x @ A in the backward.
Theirs is fused in Triton; this is plain PyTorch, same math.

Same ops, same order as PEFT's own forward (read from the installed 0.21.0 source):
  Linear4bit: result = base(x).clone(); out = lora_B(lora_A(x.to(fp32))) * s;
              result + out.to(result.dtype)
  Linear:     result = base(x); (result + lora_B(lora_A(x.to(fp32))) * s).to(result.dtype)
so the forward is bitwise the same, and the backward runs the same matmuls PEFT's autograd
would (gradients agree to fp32 rounding, not necessarily bitwise).

Only the plain path is replaced: one active adapter, no DoRA or other variant, dropout
Identity, no bias on lora_B, not merged, adapters enabled, input casting on, autocast off,
no extra forward arguments. Anything else calls PEFT's own forward and is counted as a
fallback.

  python3 lean_lora.py            CPU check (and GPU, if present) against PEFT itself
"""
import torch
import torch.nn.functional as F
from peft.tuners.lora import bnb as lora_bnb
from peft.tuners.lora import layer as lora_layer

_orig = {"Linear4bit": lora_bnb.Linear4bit.forward, "Linear": lora_layer.Linear.forward}
state = {"calls_4bit": 0, "calls_linear": 0, "fallback": 0, "installed": False}


class LoraBranch(torch.autograd.Function):
    """out = (x.to(A.dtype) @ A^T @ B^T) * scaling, saving x (not its fp32 copy)."""

    @staticmethod
    def forward(ctx, x, A, B, scaling):
        xf = x.to(A.dtype)
        out = F.linear(F.linear(xf, A), B) * scaling
        ctx.save_for_backward(x, A, B)
        ctx.scaling = scaling
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, A, B = ctx.saved_tensors
        xf = x.to(A.dtype).reshape(-1, x.shape[-1])
        a = F.linear(xf, A)  # recompute the A-activation
        g = (grad_out * ctx.scaling).reshape(-1, grad_out.shape[-1])
        grad_B = g.t().mm(a) if ctx.needs_input_grad[2] else None
        ga = g.mm(B)
        grad_A = ga.t().mm(xf) if ctx.needs_input_grad[1] else None
        grad_x = ga.mm(A).to(x.dtype).view(x.shape) if ctx.needs_input_grad[0] else None
        return grad_x, grad_A, grad_B, None


def _plain(self, args, kwargs):
    if args or kwargs or self.disable_adapters or self.merged or torch.is_autocast_enabled():
        return None
    if len(self.active_adapters) != 1 or not getattr(self, "cast_input_dtype_enabled", True):
        return None
    name = self.active_adapters[0]
    if name not in self.lora_A.keys() or name in self.lora_variant:
        return None
    if not isinstance(self.lora_dropout[name], torch.nn.Identity) or self.lora_B[name].bias is not None:
        return None
    return name


def forward_4bit(self, x, *args, **kwargs):
    name = _plain(self, args, kwargs)
    if name is None:
        state["fallback"] += 1
        return _orig["Linear4bit"](self, x, *args, **kwargs)
    self._check_forward_args(x)
    state["calls_4bit"] += 1
    result = self.base_layer(x).clone()
    out = LoraBranch.apply(x, self.lora_A[name].weight, self.lora_B[name].weight, self.scaling[name])
    return result + out.to(result.dtype)


def forward_linear(self, x, *args, **kwargs):
    name = _plain(self, args, kwargs)
    if name is None:
        state["fallback"] += 1
        return _orig["Linear"](self, x, *args, **kwargs)
    self._check_forward_args(x)
    state["calls_linear"] += 1
    result = self.base_layer(x)
    out = LoraBranch.apply(x, self.lora_A[name].weight, self.lora_B[name].weight, self.scaling[name])
    return (result + out).to(result.dtype)


def install():
    lora_bnb.Linear4bit.forward = forward_4bit
    lora_layer.Linear.forward = forward_linear
    state["installed"] = True


def uninstall():
    lora_bnb.Linear4bit.forward = _orig["Linear4bit"]
    lora_layer.Linear.forward = _orig["Linear"]
    state["installed"] = False


def report():
    return dict(state, installed=lora_bnb.Linear4bit.forward is forward_4bit
                and lora_layer.Linear.forward is forward_linear)


if __name__ == "__main__":
    import json

    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model

    def saved_bytes(fn):
        """Bytes the autograd graph saves for fn(), unique storages, parameters excluded."""
        seen = {}

        def pack(t):
            if not isinstance(t, torch.nn.Parameter):
                seen[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out = fn()
        return out, sum(seen.values())

    def check(device, n, d_in, d_out, kind):
        torch.manual_seed(0)
        if kind == "Linear4bit":
            base = bnb.nn.Linear4bit(d_in, d_out, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4")
            base.weight = bnb.nn.Params4bit(torch.randn(d_out, d_in) * 0.02, requires_grad=False, quant_type="nf4")
            base = base.to(device)
        else:
            base = torch.nn.Linear(d_in, d_out, bias=False).to(device, torch.bfloat16)
        net = torch.nn.Sequential()
        net.add_module("proj", base)
        m = get_peft_model(net, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["proj"]))
        lora = m.base_model.model.proj
        with torch.no_grad():  # B = 0 at init would hide A's gradient
            lora.lora_B["default"].weight.normal_(0, 0.02)
        x0 = torch.randn(1, n, d_in, device=device, dtype=torch.bfloat16)
        go = torch.randn(1, n, d_out, device=device, dtype=torch.bfloat16)
        out = {}
        for mode in ("peft", "lean"):
            (install if mode == "lean" else uninstall)()
            x = x0.clone().requires_grad_(True)
            y, nb = saved_bytes(lambda: m(x))
            y.backward(go)
            A, B = lora.lora_A["default"].weight, lora.lora_B["default"].weight
            out[mode] = {"y": y.detach(), "gx": x.grad, "gA": A.grad.clone(), "gB": B.grad.clone(), "saved": nb}
            A.grad = B.grad = None
        uninstall()
        p, q = out["peft"], out["lean"]

        def rel(a, b):
            return float((a.float() - b.float()).abs().max() / b.float().abs().max())

        return {"device": device, "kind": kind, "tokens": n, "in": d_in, "out": d_out,
                "adapter_dtype": str(lora.lora_A["default"].weight.dtype),
                "forward_bitwise_equal": bool(torch.equal(p["y"], q["y"])),
                "grad_x_rel_max_diff": rel(q["gx"], p["gx"]), "grad_x_bitwise_equal": bool(torch.equal(p["gx"], q["gx"])),
                "grad_A_rel_max_diff": rel(q["gA"], p["gA"]), "grad_B_rel_max_diff": rel(q["gB"], p["gB"]),
                "saved_MiB_peft": round(p["saved"] / 2**20, 3), "saved_MiB_lean": round(q["saved"] / 2**20, 3)}

    rows = []
    for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        n = 2048 if dev == "cuda" else 64
        for kind in ("Linear", "Linear4bit"):
            try:
                rows.append(check(dev, n, 2688, 3712, kind))
            except Exception as e:
                rows.append({"device": dev, "kind": kind, "error": f"{type(e).__name__}: {str(e)[:300]}"})
    print(json.dumps(rows, indent=1))

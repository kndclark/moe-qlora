"""_lora_fix checks on the CPU: with ids remapped for Marlin (hotcold's permutation), the MoE
LoRA call must still get the router's ids, Marlin the remapped ones, and nothing may leak
past the layer (also when it raises). Needs vLLM importable (run in the vLLM image).
usage: python3 lora_fix_test.py"""
import os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_kstage as ks  # noqa: E402
from vllm.model_executor.layers.fused_moe.experts import lora_experts_mixin as lem  # noqa: E402
from vllm.model_executor.layers.fused_moe.experts import marlin_moe  # noqa: E402

M = lem.LoRAExpertsMixin
assert "apply_w13_lora" not in vars(marlin_moe.MarlinExperts), "Marlin overrides apply_w13_lora"
seen = {}


def recorder(self, lora_context, **kw):
    seen["lora"] = kw["topk_ids"].clone()
    return "aligned"


M.apply_w13_lora = recorder
ks._lora_fix(M)
wrapped = M.apply_w13_lora
ks._lora_fix(M)  # twice is once
assert M.apply_w13_lora is wrapped and wrapped._kstage


class Experts(M):  # what Marlin does inside forward_modular
    pass


class Routed:
    def __init__(self, fail=False):
        self.experts, self.fail = Experts(), fail

    def forward_modular(self, x, topk_weights, topk_ids):
        seen["marlin"] = topk_ids.clone()
        assert self.experts.apply_w13_lora("ctx", y=x, topk_ids=topk_ids) == "aligned"
        if self.fail:
            raise RuntimeError("boom")
        return x


class Runner:
    def __init__(self, fail=False):
        self.routed_experts = Routed(fail)


E = 8
perm = list(reversed(range(E)))
pos = torch.empty(E, dtype=torch.long)
pos[torch.tensor(perm)] = torch.arange(E)
ids = torch.tensor([[0, 3], [7, 5]], dtype=torch.int32)
for fail in (False, True):
    r = Runner(fail)
    ks._wrap(r, 1, E, pos=pos)
    try:
        r.routed_experts.forward_modular(torch.zeros(2, 4), torch.ones(2, 2), ids)
    except RuntimeError:
        assert fail
    assert torch.equal(seen["lora"], ids), (seen["lora"], ids)
    assert torch.equal(seen["marlin"], pos[ids.long()].to(ids.dtype)), seen["marlin"]
    assert not torch.equal(seen["marlin"], ids)
    assert ks._lora_ids[0] is None, "router ids leaked past the layer"
# outside a wrapped layer the call is untouched
other = torch.tensor([[1, 2]], dtype=torch.int32)
Experts().apply_w13_lora("ctx", topk_ids=other)
assert torch.equal(seen["lora"], other)
print("lora fix: LoRA got the router's ids, Marlin the remapped ones, nothing leaked; all checks passed")

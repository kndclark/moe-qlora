"""Route 1 with attention q/k/v/o_proj kept in bf16: a wrapper around any probe.

A new configuration (David authorised follow-up runs 2026-09-25), recorded beside the gate
results, never in place of them. G2 found the NF4 attention projections to be the largest
per-layer error (laptop, isolated update error mean 0.227, max 0.457; mamba 0.127, MoE
0.107), so this keeps the 6 attention layers' q/k/v/o_proj in bf16 and changes nothing
else. Same pass lines; the Qwen3-8B yardstick stays qlora.py's NF4 config (every linear
layer 4-bit), so G2 asks whether this Lightning load is within 2x of what the lab uses now.
Cost (ARITHMETIC, config.json): 6 x 23,396,352 params = 0.261 GiB bf16 vs 0.067 GiB NF4.

How: route1.load() builds its BitsAndBytesConfig from route1's own namespace, so this
swaps that name for one that adds llm_int8_skip_modules. transformers 5.16.1 skips a
Linear whose full name ends with a listed key (quantizers_utils.should_convert_module),
and an explicit list replaces the default skips (base.get_modules_to_not_convert), so
lm_head is listed again. route1.py and the probes run unchanged, byte for byte.

Wraps a probe or a lean-scan wrapper (the first argument is the script in probes/):
  ... /probes/attn_bf16.py g2_fidelity.py lightning LABEL
  ... /probes/attn_bf16.py g2_lean_scan.py lightning LABEL
  ... /probes/attn_bf16.py g5_lean_scan.py LABEL 512,1024,2048
After the probe exits, its JSON gains an "attn_bf16" block: the skip list and what the
loaded model holds (24 bf16 attention projections and 0 Linear4bit ones expected).
"""
import functools
import json
import os
import runpy
import sys

import bitsandbytes as bnb
import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import route1  # noqa: E402

SKIP = ["mixer.q_proj", "mixer.k_proj", "mixer.v_proj", "mixer.o_proj", "lm_head"]
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
script, args = sys.argv[1], sys.argv[2:]
if script.startswith("g2"):
    assert args[0] == "lightning", "attention-bf16 G2 is lightning mode only"
    label = args[1]
else:
    label = args[0]
report = {"skip_modules": SKIP}

route1.BitsAndBytesConfig = functools.partial(route1.BitsAndBytesConfig, llm_int8_skip_modules=SKIP)
_load = route1.load


def load():
    model, info, seconds = _load()
    attn = [(n, m) for n, m in model.named_modules() if n.rpartition(".")[2] in ATTN]
    report.update({
        "attn_linear4bit": sum(isinstance(m, bnb.nn.Linear4bit) for _, m in attn),
        "attn_bf16_linear": sum(type(m) is torch.nn.Linear and m.weight.dtype == torch.bfloat16 for _, m in attn),
        "n_linear4bit_total": sum(isinstance(m, bnb.nn.Linear4bit) for m in model.modules()),
        "lm_head": [type(model.lm_head).__name__, str(model.lm_head.weight.dtype)],
    })
    return model, info, seconds


route1.load = load
sys.argv = [os.path.join(here, script)] + args
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    path = f"/out/{label}.json"
    if os.path.exists(path):
        res = json.load(open(path))
        res["attn_bf16"] = report
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
    print("attn_bf16", json.dumps(report))

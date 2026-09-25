"""Does the lean scan move G5's step-0 loss, or is step 0 simply not repeatable?

G5 step 0 (LoRA B = 0, so the frozen Route 1 model) gave loss 3.1572 unpatched and
3.1861 with lean_scan on the same 3090, and 3.1907 unpatched on the laptop. This loads
the model once, set up as g5_train_step.py does (PEFT, gradient checkpointing, train
mode), and computes the loss on the same first 512 tokens five times: original,
original, lean, lean, original. Also the max |logit difference| and the number of
positions whose argmax token changes, each against the first run.

  originals disagree with each other      -> step 0 is not repeatable; the scan is not it
  originals agree, lean runs differ       -> the lean scan's differences are amplified

Run as g5_train_step.py, with this script and a LABEL (desktop: add --memory=24g).
"""
import json
import os
import random
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_scan  # noqa: E402
import route1  # noqa: E402
from route1 import REPO, REV  # noqa: E402

sys.path.insert(0, "/gpulab/training")
from tools import TOOLS  # noqa: E402

label = sys.argv[1]
L = 512
TARGETS = r".*\.mixer\.(q_proj|k_proj|v_proj|o_proj|in_proj)$|.*\.mixer\.shared_experts\.(up_proj|down_proj)$"
assert lean_scan.modeling_sha256() == lean_scan.EXPECTED_SHA256

# the first L tokens of g5_train_step.py's stream: same shuffle, template and TOOLS
tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
records = json.load(open("/gpulab/training/research_dataset_v3.json"))
random.Random(0).shuffle(records)
stream = []
for r in records:
    msgs = []
    for m in r["messages"]:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                c = json.loads(json.dumps(c))
                fn = c.get("function", c)
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"])
                calls.append(c)
            m["tool_calls"] = calls
        msgs.append(m)
    text = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False)
    stream.extend(tok(text, add_special_tokens=False)["input_ids"])
    if len(stream) >= L:
        break
x = torch.tensor([stream[:L]], device=0)

model, info, load_s = route1.load()
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                                         task_type="CAUSAL_LM", target_modules=TARGETS))
model.train()

nh = lean_scan.nh
runs, first = [], None
for which in ("original", "original", "lean", "lean", "original"):
    nh.mamba2_chunk_scan = lean_scan.ORIGINAL if which == "original" else lean_scan.mamba2_chunk_scan
    calls0 = lean_scan.calls
    out = model(input_ids=x, labels=x, use_cache=False)
    logits = out.logits.detach().float()
    row = {"scan": which, "loss": out.loss.item(), "lean_calls": lean_scan.calls - calls0}
    if first is None:
        first = logits
    else:
        row["max_abs_logit_diff_vs_run0"] = (logits - first).abs().max().item()
        row["argmax_changed_vs_run0"] = int((logits.argmax(-1) != first.argmax(-1)).sum().item())
    runs.append(row)
    del out, logits
    torch.cuda.empty_cache()
    print(json.dumps(row))

res = {"label": label, "gpu": torch.cuda.get_device_name(), "seq": L, "first_tokens": stream[:8],
       "missing_keys": len(info["missing_keys"]), "load_seconds": round(load_s, 1), "runs": runs}
with open(f"/out/{label}.json", "w") as f:
    json.dump(res, f, indent=1)

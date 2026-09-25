"""Which scan is closer to exact: the 5.16.1 original or lean_scan? An fp64 reference.

lean-scan-layer-check-desktop.json: per call, lean and original agree to <= 4.3e-6 on
real inputs, yet the step-0 loss moves 3.157189 -> 3.186051, and Gaussian noise of the
same std moves it only 0.003-0.007. So the rounding differs in structure, and the loss
is sensitive to it. This builds `scan64` from lean_scan's source with every `.float()`
made `.double()` (the scan runs in fp64; its output is cast back to fp32 as the others
return), then, in one model load set up as g5_train_step.py does:

  fp64      every call uses scan64: the step-0 loss with a near-exact scan.
  compare   every call runs original, lean and scan64 on the same inputs and records
            each one's max |x - scan64| / max |scan64|; returns scan64's output.

Run as lean_scan_loss_check.py.
"""
import inspect
import json
import os
import random
import sys

import torch
import torch.nn.functional as F
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

src = inspect.getsource(lean_scan.mamba2_chunk_scan)
assert src.count(".float()") == 5, src.count(".float()")
ns = {"torch": torch, "F": F, "nh": lean_scan.nh, "calls": 0}
exec(src.replace(".float()", ".double()").replace("def mamba2_chunk_scan(", "def scan64_raw("), ns)


def scan64(*args, **kwargs):
    r = ns["scan64_raw"](*args, **kwargs)
    return tuple(t.float() for t in r) if isinstance(r, tuple) else r.float()


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

per_call = []
mode = {"name": None}


def first(r):
    return r[0] if isinstance(r, tuple) else r


def wrapper(*args, **kwargs):
    with torch.no_grad():
        ref = scan64(*args, **kwargs)
        if mode["name"] == "compare":
            r64 = first(ref).float()
            scale = r64.abs().max().item()
            row = {"call": len(per_call), "max_abs_ref64": scale}
            for name, fn in (("original", lean_scan.ORIGINAL), ("lean", lean_scan.mamba2_chunk_scan)):
                d = (first(fn(*args, **kwargs)).float() - r64).abs()
                row[f"{name}_rel"] = d.max().item() / scale
                row[f"{name}_mean_abs"] = d.mean().item()
            per_call.append(row)
        return ref


nh.mamba2_chunk_scan = wrapper
runs = []
for name in ("fp64", "compare"):
    mode["name"] = name
    loss = model(input_ids=x, labels=x, use_cache=False).loss.item()  # grad on, as the loss check ran
    runs.append({"mode": name, "loss": loss})
    print(json.dumps(runs[-1]))

res = {"label": label, "gpu": torch.cuda.get_device_name(), "seq": L, "first_tokens": stream[:8],
       "reference_losses": {"original": 3.157188892364502, "lean": 3.186051368713379},
       "runs": runs, "per_call": per_call,
       "lean_closer_calls": sum(c["lean_mean_abs"] < c["original_mean_abs"] for c in per_call),
       "n_calls": len(per_call),
       "sum_mean_abs": {k: sum(c[f"{k}_mean_abs"] for c in per_call) for k in ("original", "lean")},
       "worst_rel": {k: max(c[f"{k}_rel"] for c in per_call) for k in ("original", "lean")}}
with open(f"/out/{label}.json", "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps({k: res[k] for k in ("lean_closer_calls", "n_calls", "sum_mean_abs", "worst_rel")}))

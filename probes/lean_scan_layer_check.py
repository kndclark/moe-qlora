"""Is the lean scan wrong on real inputs, or does the model amplify tiny differences?

lean-scan-loss-check-desktop.json: step-0 loss is exactly repeatable (original 3.157189
twice, lean 3.186051 twice), yet the random-input check agreed to ~1e-7. Two tests, one
model load (set up as g5_train_step.py does), the same first 512 tokens:

  compare   each of the 23 scan calls runs original and lean on the same real inputs;
            records max |lean - original| / max |original| per call and returns the
            original's output, so the loss must equal 3.157189 (the wrapper is inert).
  noise s   returns original + Gaussian noise whose std equals that call's measured
            std(lean - original) (seeds 1 and 2). If the loss moves by a similar amount,
            the model amplifies any perturbation that small; the lean scan is not special.

Run as lean_scan_loss_check.py.
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

per_call, noise_std = [], []
mode = {"name": None, "gen": None, "i": 0}


def wrapper(*args, **kwargs):
    with torch.no_grad():
        ref = lean_scan.ORIGINAL(*args, **kwargs)
        ref_out = ref[0] if isinstance(ref, tuple) else ref
        if mode["name"] == "compare":
            new = lean_scan.mamba2_chunk_scan(*args, **kwargs)
            new_out = new[0] if isinstance(new, tuple) else new
            d = (new_out.float() - ref_out.float())
            scale = ref_out.float().abs().max().item()
            per_call.append({"call": len(per_call), "dtype": str(ref_out.dtype), "max_abs_ref": scale,
                             "max_abs_diff": d.abs().max().item(), "rel": d.abs().max().item() / scale,
                             "diff_std": d.std().item(), "tuple": isinstance(ref, tuple),
                             "kwargs": sorted(k for k, v in kwargs.items() if v is not None)})
            noise_std.append(d.std().item())
            return ref
        g = mode["gen"]
        std = noise_std[mode["i"] % len(noise_std)]
        mode["i"] += 1
        noise = torch.randn(ref_out.shape, generator=g, device="cpu").to(ref_out.device) * std
        out = (ref_out.float() + noise).to(ref_out.dtype)
        return (out,) + tuple(ref[1:]) if isinstance(ref, tuple) else out


nh.mamba2_chunk_scan = wrapper
runs = []
for name, seed in (("compare", None), ("noise", 1), ("noise", 2)):
    mode.update(name=name, gen=torch.Generator().manual_seed(seed) if seed else None, i=0)
    loss = model(input_ids=x, labels=x, use_cache=False).loss.item()  # grad on, as the loss check ran
    runs.append({"mode": name, "seed": seed, "loss": loss})
    print(json.dumps(runs[-1]))

res = {"label": label, "gpu": torch.cuda.get_device_name(), "seq": L, "first_tokens": stream[:8],
       "reference_losses": {"original": 3.157188892364502, "lean": 3.186051368713379},
       "runs": runs, "per_call": per_call,
       "worst_rel": max(c["rel"] for c in per_call), "n_calls": len(per_call)}
with open(f"/out/{label}.json", "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps({"worst_rel": res["worst_rel"], "n_calls": res["n_calls"],
                  "rels": [f"{c['rel']:.1e}" for c in per_call]}))

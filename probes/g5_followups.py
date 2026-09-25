"""F2 and F3 (plan.md): g5_train_step.py, byte for byte, on the F1 configuration (lean_scan +
chunked_ce) plus, by environment switch:
  LEAN_LORA=1                   F2: lean_lora.py's backward for PEFT Linear4bit / Linear
  EXPERT_LORA=shared|per_expert F3: expert_lora.py on the routed experts (EXPERT_R, default 16;
                                alpha = 2 r; shared factors fp32, per-expert factors bf16)
  LM_HEAD_LORA=1                F3: lm_head added to g5_train_step's PEFT target regex
Nothing set = the F1 run. The expert factors are attached inside get_peft_model (before
PEFT's own, which freezes them, then re-enabled), so g5_train_step's optimizer and its
trainable-parameter count include them. After the probe exits, its JSON gains "lean_scan",
"chunked_ce", "followups" and, when used, "lean_lora" / "expert_lora" blocks.

  ... /probes/attn_bf16.py g5_followups.py LABEL [512,1024,2048]
"""
import json
import os
import runpy
import sys

import peft
import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import chunked_ce  # noqa: E402
import lean_scan  # noqa: E402

LEAN = os.environ.get("LEAN_LORA") == "1"
MODE = os.environ.get("EXPERT_LORA", "none")
R = int(os.environ.get("EXPERT_R", "16"))
LM_HEAD = os.environ.get("LM_HEAD_LORA") == "1"
assert MODE in ("none", "shared", "per_expert"), MODE
lean_scan.install()
chunked_ce.install(int(os.environ.get("CE_CHUNK", "256")))
if LEAN:
    import lean_lora
    lean_lora.install()
if MODE != "none":
    import expert_lora
follow = {"lean_lora": LEAN, "expert_lora": MODE, "expert_r": R if MODE != "none" else None,
          "lm_head_lora": LM_HEAD}
lm_track = []
_get_peft_model = peft.get_peft_model


def get_peft_model(model, config, *args, **kwargs):
    params = []
    if MODE != "none":
        params = expert_lora.attach_expert_lora(model, r=R, alpha=2 * R, mode=MODE,
                                                dtype=torch.float32 if MODE == "shared" else torch.bfloat16)
    if LM_HEAD:
        config.target_modules = config.target_modules + "|lm_head"
    follow["target_modules"] = config.target_modules
    model = _get_peft_model(model, config, *args, **kwargs)
    for p in params:
        p.requires_grad_(True)
    for n, p in model.named_parameters():
        if "lm_head.lora_" in n:
            pos = torch.linspace(0, p.numel() - 1, 256, device=p.device).long()
            lm_track.append((n, p, pos, p.detach().flatten()[pos].clone()))
    follow["lm_head_lora_params"] = sum(p.numel() for _, p, _, _ in lm_track)
    return model


peft.get_peft_model = get_peft_model
label = sys.argv[1]
sys.argv = [os.path.join(here, "g5_train_step.py")] + sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    follow["lm_head_factors_moved"] = {n.split("lm_head.")[1]: not torch.equal(p.detach().flatten()[pos], init)
                                       for n, p, pos, init in lm_track}
    blocks = {"lean_scan": lean_scan.report(), "chunked_ce": chunked_ce.report(), "followups": follow}
    if LEAN:
        blocks["lean_lora"] = lean_lora.report()
    if MODE != "none":
        blocks["expert_lora"] = expert_lora.report()
    path = f"/out/{label}.json"
    if os.path.exists(path):
        res = json.load(open(path))
        res.update(blocks)
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
    print(json.dumps(blocks, default=str))

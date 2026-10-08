# How much does vLLM's B12x path lose by folding each expert's global weight scale
# (weight_scale_2) into its FP8 E4M3 block scales? CPU only, reads the checkpoint.
# Run (CPU, reads only scales): docker run --rm --pull never -v /srv/model-cache:/hf:ro
#   -v $PWD/probes:/p --entrypoint python3 vllm/vllm-openai:v0.29.0 /p/b12x_scale_bake.py
import json, os, glob, torch
from safetensors import safe_open
d = glob.glob("/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/*/")[0]
wm = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
ex = sorted({k.rsplit(".experts.", 1)[0] for k in wm if ".experts." in k})
print("moe layers", len(ex), "e.g.", ex[0])
names = [k for k in wm if k.startswith(ex[0] + ".experts.0.")]
print(names)
def get(k):
    with safe_open(d + wm[k], "pt") as f: return f.get_tensor(k)
for L in ex[:: max(1, len(ex) // 6)]:
    for proj in ("up_proj", "down_proj"):
        tot = z = 0; rel = []; s2s = []
        for e in range(0, 128, 8):
            p = f"{L}.experts.{e}.{proj}."
            s = get(p + "weight_scale").float(); s2 = get(p + "weight_scale_2").float()
            true = s * s2; baked = true.to(torch.float8_e4m3fn).float()
            nz = true > 0; tot += nz.sum().item(); z += (nz & (baked == 0)).sum().item()
            rel.append(((baked - true).abs() / true.clamp_min(1e-30))[nz]); s2s.append(s2.item())
        r = torch.cat(rel)
        print(f"{L.split('layers.')[1]:>16} {proj:9} s2 {min(s2s):.2e}..{max(s2s):.2e}  zeroed {z/tot:.4%}  "
              f"rel err median {r.median():.3%} p99 {r.quantile(.99):.2%} >25% {(r>.25).float().mean():.3%}")

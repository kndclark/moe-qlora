"""G4, route 1: load Lightning BF16 onto one GPU with 4-bit experts; measure what stays resident.

Route 1 = transformers' built-in nemotron_h class, BitsAndBytesConfig NF4 for
every nn.Linear, and bnb 0.50.2 replace_parameter_4bit on the fused 3D expert
Parameters as each one is loaded. The load-time hook is adapted from axolotl
0.19.0 src/axolotl/monkeypatch/moe_quant.py (patch_moe_quantization_on_load):
4-bit only, no axolotl dependency. Why a hook: the experts are 54.7 GiB in
bf16, so each fused stack must be quantized as soon as it lands on the GPU.

Facts this relies on (read in the image, transformers 5.16.1):
  - core_model_loading calls set_param_for_module as a module global, so
    patching the module attribute takes effect.
  - With on-the-fly quantization the loader runs without a thread pool (one
    tensor at a time), and materialize_tensors pops each expert's source
    tensors before the next merge.
  - The nemotron_h conversion merges mixer.experts.*.{up,down}_proj.weight into
    one (128, out, in) Parameter per layer on the target device (no force_cpu).
  - NemotronHPreTrainedModel ignores the 270 mtp.* tensors.

Measures (GiB = 2**30 bytes):
  load wall time; host peak RSS; torch peak allocated/reserved during load;
  device-wide used memory from NVML, sampled every 0.25 s (other apps included,
  so the pre-CUDA baseline is recorded too); resident bytes by storage class;
  expert Parameters left unquantized (must be 0); missing/unexpected keys;
  then a sanity forward (use_cache=False): mean NLL on a fixed paragraph and
  top-1 next token on three cloze prompts, with its peak VRAM. The sanity
  forward catches a garbled load. It is not the fidelity gate (that is G2).

Run on the laptop (download G0 first):
  docker run --rm --init --gpus all --ipc=host -v /srv/model-cache:/hf:ro \
    -v ~/moe-qlora/probes:/probes:ro -v ~/moe-qlora/results:/out \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e PYTHONDONTWRITEBYTECODE=1 --user $(id -u):$(id -g) \
    --entrypoint python3 gpu-lab:training /probes/g4_route1_load.py LABEL
HF_HUB_OFFLINE=1 also keeps the kernels hub from downloading, so mamba runs
its torch path; the JSON records that.
"""
import gc
import json
import os
import resource
import sys
import threading
import time

import bitsandbytes as bnb
import peft
import pynvml
import torch
import transformers
from bitsandbytes.nn.parametrize import replace_parameter_4bit
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
REV = "a9904d24bcc1d289a1950fa9d2b978c47cf903b9"
GiB = 2**30
label = sys.argv[1] if len(sys.argv) > 1 else "g4-route1"
# Smoke-test overrides: a tiny local nemotron_h checkpoint and any local tokenizer.
MODEL = os.environ.get("G4_MODEL_DIR", REPO)
TOKENIZER = os.environ.get("G4_TOKENIZER", REPO)
rev = None if "G4_MODEL_DIR" in os.environ else REV
tok_rev = None if "G4_TOKENIZER" in os.environ else REV


def gib(n):
    return round(n / GiB, 3)


# NVML sampler: device-wide used memory, temperature, host RSS. Peaks per phase.
pynvml.nvmlInit()
nvh = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get("NVML_INDEX", "0")))


def rss_bytes():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.phase = "init"
        self.peaks = {}
        self.stop = False

    def sample(self):
        used = pynvml.nvmlDeviceGetMemoryInfo(nvh).used
        temp = pynvml.nvmlDeviceGetTemperature(nvh, pynvml.NVML_TEMPERATURE_GPU)
        p = self.peaks.setdefault(self.phase, {"nvml_used": 0, "temp_C": 0, "rss": 0})
        p["nvml_used"] = max(p["nvml_used"], used)
        p["temp_C"] = max(p["temp_C"], temp)
        p["rss"] = max(p["rss"], rss_bytes())
        return used

    def run(self):
        while not self.stop:
            self.sample()
            time.sleep(0.25)


baseline_nvml_used = pynvml.nvmlDeviceGetMemoryInfo(nvh).used  # before this process touches CUDA
sampler = Sampler()
sampler.start()

# Load-time hook (after axolotl 0.19.0 patch_moe_quantization_on_load, 4-bit branch).
import transformers.core_model_loading as cml  # noqa: E402
import transformers.modeling_utils as mu  # noqa: E402

hook = {"quantized": [], "skipped_not_cuda": [], "skipped_no_expert_in_name": []}
_orig_set_param = cml.set_param_for_module


def _set_param_then_quantize(model, target_name, param_value, *args, **kwargs):
    _orig_set_param(model, target_name, param_value, *args, **kwargs)
    if param_value.ndim < 3:
        return
    mod_path, _, pname = target_name.rpartition(".")
    mod = model.get_submodule(mod_path) if mod_path else model
    if isinstance(mod, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
        return
    if "expert" not in target_name.lower():
        hook["skipped_no_expert_in_name"].append([target_name, list(param_value.shape)])
        return
    if not param_value.is_cuda:
        hook["skipped_not_cuda"].append([target_name, list(param_value.shape)])
        return
    replace_parameter_4bit(mod, pname, compress_statistics=True, quant_type="nf4")
    hook["quantized"].append(target_name)
    param_value.data = torch.empty(0, device="cpu")  # free the bf16 stack now
    torch.cuda.empty_cache()


cml.set_param_for_module = _set_param_then_quantize
# caching_allocator_warmup pre-allocates at bf16 size for every param, which
# defeats quantize-on-load (axolotl disables it for the same reason).
mu.caching_allocator_warmup = lambda *a, **k: None

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

torch.cuda.init()
cuda_ctx_nvml_used = sampler.sample()
sampler.phase = "load"
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
model, info = AutoModelForCausalLM.from_pretrained(
    MODEL, revision=rev, quantization_config=bnb_cfg, dtype=torch.bfloat16,
    device_map={"": 0}, output_loading_info=True,
)
load_s = time.time() - t0
load_peak_alloc = torch.cuda.max_memory_allocated()
load_peak_reserved = torch.cuda.max_memory_reserved()
host_peak_rss_ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

# Idle: what the loaded model holds at rest.
sampler.phase = "idle"
gc.collect()
torch.cuda.empty_cache()
time.sleep(3)
idle = {"torch_allocated_GiB": gib(torch.cuda.memory_allocated()),
        "torch_reserved_GiB": gib(torch.cuda.memory_reserved()),
        "nvml_used_GiB": gib(sampler.sample())}


# Resident bytes by storage class.
def qs_bytes(qs):
    n = 0
    for t in (qs.absmax, qs.code, getattr(qs, "offset", None)):
        if isinstance(t, torch.Tensor):
            n += t.numel() * t.element_size()
    if getattr(qs, "state2", None) is not None:
        for t in (qs.state2.absmax, qs.state2.code):
            if isinstance(t, torch.Tensor):
                n += t.numel() * t.element_size()
    return n


inv = {"expert_packed": 0, "expert_stats": 0, "expert_params_4bit": 0,
       "linear4bit_packed": 0, "linear4bit_stats": 0, "linear4bit_params": 0, "linear4bit_modules": 0,
       "other_bytes": 0, "other_params": 0}
other_by_dtype = {}
unquantized_3d_expert = []
seen = set()
for mname, mod in model.named_modules():
    if hasattr(mod, "parametrizations"):
        for pname, plist in mod.parametrizations.items():
            orig = plist.original
            seen.add(id(orig))
            inv["expert_packed"] += orig.numel() * orig.element_size()
            inv["expert_stats"] += qs_bytes(plist[0].quant_state)
            inv["expert_params_4bit"] += plist[0].quant_state.shape.numel()
    elif isinstance(mod, bnb.nn.Linear4bit):
        w = mod.weight
        seen.add(id(w))
        inv["linear4bit_modules"] += 1
        inv["linear4bit_packed"] += w.numel() * w.element_size()
        inv["linear4bit_stats"] += qs_bytes(w.quant_state)
        inv["linear4bit_params"] += w.quant_state.shape.numel()
for n, t in list(model.named_parameters()) + list(model.named_buffers()):
    if id(t) in seen:
        continue
    seen.add(id(t))
    b = t.numel() * t.element_size()
    inv["other_bytes"] += b
    inv["other_params"] += t.numel()
    other_by_dtype[str(t.dtype)] = other_by_dtype.get(str(t.dtype), 0) + b
    if t.ndim >= 3 and "expert" in n:
        unquantized_3d_expert.append([n, list(t.shape), str(t.dtype)])
inventory = {k: (gib(v) if k.endswith(("packed", "stats", "bytes")) else v) for k, v in inv.items()}
inventory["other_by_dtype_GiB"] = {k: gib(v) for k, v in other_by_dtype.items()}
inventory["sum_GiB"] = gib(inv["expert_packed"] + inv["expert_stats"] + inv["linear4bit_packed"]
                           + inv["linear4bit_stats"] + inv["other_bytes"])

# Sanity forward.
sampler.phase = "forward"
tok = AutoTokenizer.from_pretrained(TOKENIZER, revision=tok_rev)
paragraph = (
    "The river rises in the hills north of the town and runs south for about forty miles "
    "before it reaches the sea. In the spring the water is high and fast, and the old stone "
    "bridge near the market has been closed more than once. Farmers along the valley grow "
    "wheat, barley and apples, and most of the harvest is sold at the weekly market in the "
    "square. The town has a small museum, two schools and a railway station that opened in 1872."
)
cloze = ["The capital of France is", "The chemical symbol for gold is", "One, two, three, four,"]
model.eval()
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    enc = tok(paragraph, return_tensors="pt").to(0)
    out = model(**enc, labels=enc["input_ids"], use_cache=False)
    nll = out.loss.item()
    finite = bool(torch.isfinite(out.logits).all())
    top1 = {}
    for p in cloze:
        e = tok(p, return_tensors="pt").to(0)
        top1[p] = tok.decode(model(**e, use_cache=False).logits[0, -1].argmax().item())
fwd_peak_alloc = torch.cuda.max_memory_allocated()
sampler.stop = True
time.sleep(0.3)

# transformers resolves mamba ops as: hub kernels (if requested) > installed package > torch.
mamba_kernels = {"HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE")}
for pkg in ("mamba_ssm", "causal_conv1d"):
    try:
        __import__(pkg)
        mamba_kernels[pkg] = "importable"
    except Exception as e:
        mamba_kernels[pkg] = type(e).__name__

phase_peaks = {ph: {"nvml_used_GiB": gib(p["nvml_used"]), "temp_C": p["temp_C"], "rss_GiB": gib(p["rss"])}
               for ph, p in sampler.peaks.items()}
res = {
    "label": label, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "model": MODEL, "revision": rev, "tokenizer": TOKENIZER,
    "gpu": torch.cuda.get_device_name(), "gpu_total_GiB": gib(torch.cuda.get_device_properties(0).total_memory),
    "torch": torch.__version__, "transformers": transformers.__version__,
    "bitsandbytes": bnb.__version__, "peft": peft.__version__,
    "experts_implementation": getattr(model.config, "_experts_implementation", None),
    "mamba": mamba_kernels,
    "baseline_nvml_used_GiB_before_cuda": gib(baseline_nvml_used),
    "nvml_used_GiB_after_cuda_init": gib(cuda_ctx_nvml_used),
    "load": {"seconds": round(load_s, 1),
             "torch_peak_allocated_GiB": gib(load_peak_alloc),
             "torch_peak_reserved_GiB": gib(load_peak_reserved),
             "host_peak_rss_GiB_getrusage": gib(host_peak_rss_ru)},
    "idle": idle,
    "phase_peaks": phase_peaks,
    "hook": {"n_quantized": len(hook["quantized"]),
             "skipped_not_cuda": hook["skipped_not_cuda"],
             "skipped_no_expert_in_name": hook["skipped_no_expert_in_name"][:10],
             "n_skipped_no_expert_in_name": len(hook["skipped_no_expert_in_name"])},
    "inventory": inventory,
    "unquantized_3d_expert_params": unquantized_3d_expert,
    "loading_info": {"missing_keys": sorted(info["missing_keys"])[:50],
                     "n_missing": len(info["missing_keys"]),
                     "n_unexpected": len(info["unexpected_keys"]),
                     "unexpected_sample": sorted(info["unexpected_keys"])[:10],
                     "mismatched": [list(map(str, m)) for m in list(info.get("mismatched_keys", []))[:10]]},
    "sanity": {"paragraph_tokens": int(enc["input_ids"].shape[1]), "paragraph_mean_nll": round(nll, 4),
               "logits_finite": finite, "cloze_top1": top1,
               "forward_torch_peak_allocated_GiB": gib(fwd_peak_alloc)},
}
print(json.dumps(res, indent=1))
with open(f"/out/{label}.json", "w") as f:
    json.dump(res, f, indent=1)

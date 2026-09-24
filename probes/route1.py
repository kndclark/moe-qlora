"""Shared by G2 and G5: the Route 1 load that G4 measured, plus an NVML sampler.

The load-time hook is g4_route1_load.py's, copied line for line (G4's results
are the evidence that it works; importing that script would run it). See its
docstring for the transformers 5.16.1 facts it relies on.

The sampler extends G4's with the thermal guard from plan.md G5: clock-event
reasons, power and temperature every 0.25 s. `sampler.abort` is set, with a
reason, when a thermal or power-brake flag is raised or the GPU reaches 87 C
(the card's own target; memory: laptop-gpu-power-envelope). sw_power_cap is
recorded but does not abort: it is the card sitting at its power limit.
"""
import os
import threading
import time

import bitsandbytes as bnb
import pynvml
import torch
from bitsandbytes.nn.parametrize import replace_parameter_4bit
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

REPO = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
REV = "a9904d24bcc1d289a1950fa9d2b978c47cf903b9"
GiB = 2**30
ABORT_TEMP_C = 87


def gib(n):
    return round(n / GiB, 3)


def rss_bytes():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


pynvml.nvmlInit()
nvh = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get("NVML_INDEX", "0")))
REASON_BITS = {
    "sw_power_cap": pynvml.nvmlClocksEventReasonSwPowerCap,
    "sw_thermal": pynvml.nvmlClocksEventReasonSwThermalSlowdown,
    "hw_thermal": pynvml.nvmlClocksEventReasonHwThermalSlowdown,
    "hw_power_brake": pynvml.nvmlClocksEventReasonHwPowerBrakeSlowdown,
    "hw_slowdown": pynvml.nvmlClocksEventReasonHwSlowdown,
}
ABORT_REASONS = ("sw_thermal", "hw_thermal", "hw_power_brake")


class Sampler(threading.Thread):
    """Per-phase peaks of device-wide NVML used memory, temperature, power, RSS;
    counts of samples with each clock-event reason set."""

    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.phase = "init"
        self.peaks = {}
        self.stop = False
        self.abort = None

    def sample(self):
        used = pynvml.nvmlDeviceGetMemoryInfo(nvh).used
        temp = pynvml.nvmlDeviceGetTemperature(nvh, pynvml.NVML_TEMPERATURE_GPU)
        try:
            watts = pynvml.nvmlDeviceGetPowerUsage(nvh) / 1000
        except pynvml.NVMLError:
            watts = 0.0
        reasons = pynvml.nvmlDeviceGetCurrentClocksEventReasons(nvh)
        p = self.peaks.setdefault(self.phase, {"nvml_used": 0, "temp_C": 0, "watts": 0.0, "rss": 0,
                                               "samples": 0, "reasons": {k: 0 for k in REASON_BITS}})
        p["nvml_used"] = max(p["nvml_used"], used)
        p["temp_C"] = max(p["temp_C"], temp)
        p["watts"] = max(p["watts"], watts)
        p["rss"] = max(p["rss"], rss_bytes())
        p["samples"] += 1
        for k, bit in REASON_BITS.items():
            if reasons & bit:
                p["reasons"][k] += 1
                if k in ABORT_REASONS and self.abort is None:
                    self.abort = f"{k} flag set in phase {self.phase}"
        if temp >= ABORT_TEMP_C and self.abort is None:
            self.abort = f"GPU at {temp} C (limit {ABORT_TEMP_C}) in phase {self.phase}"
        return used

    def run(self):
        while not self.stop:
            self.sample()
            time.sleep(self.interval)

    def report(self):
        return {ph: {"nvml_used_GiB": gib(p["nvml_used"]), "temp_C": p["temp_C"], "watts": round(p["watts"], 1),
                     "rss_GiB": gib(p["rss"]), "samples": p["samples"],
                     "reason_samples": {k: v for k, v in p["reasons"].items() if v}}
                for ph, p in self.peaks.items()}


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


def load():
    """Load Route 1 onto cuda:0. Returns (model, loading_info, seconds)."""
    cml.set_param_for_module = _set_param_then_quantize
    # caching_allocator_warmup pre-allocates at bf16 size for every param, which
    # defeats quantize-on-load (axolotl disables it for the same reason).
    mu.caching_allocator_warmup = lambda *a, **k: None
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    t0 = time.time()
    model, info = AutoModelForCausalLM.from_pretrained(
        REPO, revision=REV, quantization_config=bnb_cfg, dtype=torch.bfloat16,
        device_map={"": 0}, output_loading_info=True,
    )
    return model, info, time.time() - t0


def hook_summary():
    return {"n_quantized": len(hook["quantized"]), "skipped_not_cuda": hook["skipped_not_cuda"],
            "n_skipped_no_expert_in_name": len(hook["skipped_no_expert_in_name"])}


def mamba_kernels():
    # transformers resolves mamba ops as: hub kernels (if requested) > installed package > torch.
    out = {"HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE")}
    for pkg in ("mamba_ssm", "causal_conv1d"):
        try:
            __import__(pkg)
            out[pkg] = "importable"
        except Exception as e:
            out[pkg] = type(e).__name__
    return out

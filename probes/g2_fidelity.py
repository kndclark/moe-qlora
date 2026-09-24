"""G2: forward fidelity of the 4-bit load against bf16 (plan.md G2).

Modes:
  lightning LABEL  bf16 reference streamed one layer at a time, then Route 1.
  qwen LABEL       yardstick: Qwen3-8B bf16 against its NF4 load (qlora.py's
                   config: nf4, double quant, bf16 compute).

Text: the first N_WIN x WIN Lightning tokens of gpu-lab docs/phase2-dev-plane.md
(the lab's own doc, written 2026, so it is in neither model's training data).
The Qwen yardstick decodes each Lightning window back to text and tokenizes it
with Qwen, so both models see the same characters, each in its own tokens.

Why 512-token windows rather than one 2k sequence: with no mamba kernels,
mamba2_chunk_scan (modeling_nemotron_h.py, torch path) materializes
C[:, :, :, None] * B[:, :, None] as fp32 (b, chunks, 128, 128, 64 heads, 128
state) before summing: 0.5 GiB per 128-token chunk, so 8 GiB at 2048 tokens,
on top of the 16.16 GiB the Route 1 model holds (ARITHMETIC). 512 tokens is
2 GiB. The same windows go through both paths, so the comparison is fair.

Lightning reference (phase A): the model class is built on the meta device and
each layer is materialized on the GPU in turn, filled from the safetensors
shards with the loader's own renames (backbone. -> model., experts.N.* stacked
into one (128, out, in) tensor), run on all windows, then sent back to meta.
Every parameter and buffer of each module must be filled from the checkpoint
(the probe fails otherwise), and the dtypes of the tensors both paths keep
unquantized are compared with Route 1's after it loads.

Route 1 (phase B), per window:
  accumulated: each block's output in the real Route 1 forward against the
               reference hidden state at the same depth;
  isolated:    each Route 1 block run on the reference input to that block;
               error of its residual update (output - input) against the
               reference update;
  end to end:  logits from Route 1 against logits from the reference final
               hidden state through the same bf16 norm_f and lm_head.
Relative error = ||a - b||_F / ||b||_F, pooled over the windows.
KL(bf16 || 4-bit) in nats per predicted token; top-1 = argmax agreement.

Run (laptop; the desktop adds --memory=24g --memory-swap=24g):
  docker run --rm --init --gpus all --ipc=host -v /srv/model-cache:/hf:ro \
    -v ~/moe-qlora/probes:/probes:ro -v ~/moe-qlora/results:/out \
    -v ~/gpu-lab/docs:/gpulab/docs:ro \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e PYTHONDONTWRITEBYTECODE=1 --user $(id -u):$(id -g) \
    --entrypoint python3 gpu-lab:training /probes/g2_fidelity.py MODE LABEL
"""
import gc
import hashlib
import json
import os
import re
import sys
import time

import bitsandbytes as bnb
import torch
import transformers
import transformers.models.nemotron_h.modeling_nemotron_h as nh
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route1  # noqa: E402
from route1 import REPO, REV, gib  # noqa: E402

MODE, label = sys.argv[1], sys.argv[2]
assert MODE in ("lightning", "qwen"), MODE
DOC = "/gpulab/docs/phase2-dev-plane.md"
N_WIN, WIN = 4, 512
QWEN = "Qwen/Qwen3-8B"
bf16 = torch.bfloat16

baseline_nvml_used = route1.pynvml.nvmlDeviceGetMemoryInfo(route1.nvh).used
sampler = route1.Sampler()
sampler.start()
torch.cuda.init()
t_start = time.time()


def windows():
    raw = open(DOC, "rb").read()
    ltok = AutoTokenizer.from_pretrained(REPO, revision=REV)
    prefix = ltok("")["input_ids"]  # whatever special tokens the tokenizer adds by default
    body = WIN - len(prefix)
    ids = ltok(raw.decode(), add_special_tokens=False)["input_ids"]
    assert len(ids) >= N_WIN * body, (len(ids), N_WIN * body)
    wins = [prefix + ids[k * body:(k + 1) * body] for k in range(N_WIN)]
    texts = [ltok.decode(ids[k * body:(k + 1) * body]) for k in range(N_WIN)]
    meta = {"doc": DOC, "doc_sha256": hashlib.sha256(raw).hexdigest(), "doc_tokens": len(ids),
            "n_windows": N_WIN, "window_tokens": WIN, "prefix_ids": prefix}
    return wins, texts, meta


class Tally:
    """End-to-end logit comparison, accumulated over windows."""

    def __init__(self):
        self.kl, self.nll_ref, self.nll_q = [], [], []
        self.top1 = 0

    @torch.no_grad()
    def add(self, logits_ref, logits_q, ids):
        lr = torch.log_softmax(logits_ref[:-1].float(), -1)
        lq = torch.log_softmax(logits_q[:-1].float(), -1)
        tgt = ids[1:, None]
        self.kl.append((lr.exp() * (lr - lq)).sum(-1).cpu())
        self.nll_ref.append(-lr.gather(-1, tgt).squeeze(-1).cpu())
        self.nll_q.append(-lq.gather(-1, tgt).squeeze(-1).cpu())
        self.top1 += int((lr.argmax(-1) == lq.argmax(-1)).sum())
        del lr, lq

    def result(self):
        kl = torch.cat(self.kl)
        nr, nq = torch.cat(self.nll_ref).mean().item(), torch.cat(self.nll_q).mean().item()
        return {"predicted_tokens": kl.numel(), "mean_kl_nats": round(kl.mean().item(), 6),
                "kl_p50": round(kl.quantile(0.5).item(), 6), "kl_p99": round(kl.quantile(0.99).item(), 6),
                "kl_max": round(kl.max().item(), 6),
                "top1_agreement_pct": round(100 * self.top1 / kl.numel(), 3),
                "nll_bf16": round(nr, 5), "nll_4bit": round(nq, 5), "nll_gap": round(nq - nr, 5)}


def dump(res):
    res.update({"label": label, "mode": MODE, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "wall_s": round(time.time() - t_start, 1),
                "gpu": torch.cuda.get_device_name(), "gpu_total_GiB": gib(torch.cuda.get_device_properties(0).total_memory),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "bitsandbytes": bnb.__version__,
                "baseline_nvml_used_GiB_before_cuda": gib(baseline_nvml_used),
                "phase_peaks": sampler.report()})
    print(json.dumps(res, indent=1))
    with open(f"/out/{label}.json", "w") as f:
        json.dump(res, f, indent=1)


def free():
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------- qwen
def run_qwen():
    wins_l, texts, meta = windows()
    qtok = AutoTokenizer.from_pretrained(QWEN)
    wins = [qtok(t)["input_ids"] for t in texts]
    snap = open(f"{os.environ['HF_HOME']}/hub/models--Qwen--Qwen3-8B/refs/main").read().strip()
    res = {"model": QWEN, "revision": snap, "text": meta, "qwen_window_tokens": [len(w) for w in wins]}

    sampler.phase = "bf16"
    m = AutoModelForCausalLM.from_pretrained(QWEN, dtype=bf16, device_map={"": 0}).eval()
    ref = []
    with torch.no_grad():
        for w in wins:
            ref.append(m(input_ids=torch.tensor([w], device=0), use_cache=False).logits[0].to(bf16).cpu())
    res["bf16_torch_peak_GiB"] = gib(torch.cuda.max_memory_allocated())
    del m
    free()

    sampler.phase = "nf4"
    torch.cuda.reset_peak_memory_stats()
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=bf16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(QWEN, quantization_config=cfg, dtype=bf16, device_map={"": 0}).eval()
    res["n_linear4bit"] = sum(isinstance(x, bnb.nn.Linear4bit) for x in m.modules())
    tally = Tally()
    with torch.no_grad():
        for w, r in zip(wins, ref):
            ids = torch.tensor([w], device=0)
            tally.add(r.to(0), m(input_ids=ids, use_cache=False).logits[0], ids[0])
    res["nf4_torch_peak_GiB"] = gib(torch.cuda.max_memory_allocated())
    res["end_to_end"] = tally.result()
    dump(res)


# ---------------------------------------------------------------- lightning
def run_lightning():
    wins, _, meta = windows()
    snap = f"{os.environ['HF_HOME']}/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/snapshots/{REV}"
    index = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
    handles = {}

    def get(key):
        shard = index[key]
        if shard not in handles:
            handles[shard] = safe_open(f"{snap}/{shard}", framework="pt", device="cpu")
        return handles[shard].get_tensor(key)

    ckpt_dtypes = {}

    def fill(module, prefix):
        """Materialize `module` on cuda:0 and fill every tensor from the checkpoint."""
        module.to_empty(device="cuda:0")
        want = dict(module.named_parameters())
        want.update(module.named_buffers())
        done, experts = set(), {}
        for key in [k for k in index if k.startswith(prefix)]:
            name = key[len(prefix):]
            t = get(key)
            m = re.fullmatch(r"mixer\.experts\.(\d+)\.(up_proj|down_proj)\.weight", name)
            if m:
                dst = want[f"mixer.experts.{m[2]}"].data[int(m[1])]
                experts.setdefault(f"mixer.experts.{m[2]}", set()).add(int(m[1]))
            else:
                dst = want[name].data  # KeyError here = a checkpoint key the class does not have
                done.add(name)
                ckpt_dtypes[prefix + name] = str(t.dtype)
            assert tuple(t.shape) == tuple(dst.shape), (key, t.shape, dst.shape)
            dst.copy_(t)
        for name, idx in experts.items():
            assert idx == set(range(128)), (prefix, name, len(idx))
            done.add(name)
        missing = set(want) - done
        assert not missing, (prefix, sorted(missing))

    # Phase A: bf16 reference, one layer on the GPU at a time.
    sampler.phase = "reference"
    cfg = AutoConfig.from_pretrained(REPO, revision=REV)
    with torch.device("meta"):
        ref = AutoModelForCausalLM.from_config(cfg, dtype=bf16)
    ref.eval()
    # from_pretrained keeps these fp32 (_keep_in_fp32_modules_strict); do the same here.
    keep32 = list(getattr(ref, "_keep_in_fp32_modules_strict", None) or []) + \
        list(getattr(ref, "_keep_in_fp32_modules", None) or [])
    for mod in ref.modules():
        for bname, b in list(mod._buffers.items()):
            if b is not None and any(k in bname for k in keep32):
                mod._buffers[bname] = b.float()
        for pname, p in list(mod._parameters.items()):
            if p is not None and any(k in pname for k in keep32):
                mod._parameters[pname] = torch.nn.Parameter(p.float(), requires_grad=False)
    ref_dtypes = {n: str(t.dtype) for n, t in list(ref.named_parameters()) + list(ref.named_buffers())}
    ref_impl = {"attn": ref.config._attn_implementation, "experts": getattr(ref.config, "_experts_implementation", None)}

    n_layers = len(ref.model.layers)
    H = torch.empty(n_layers + 1, N_WIN, WIN, cfg.hidden_size, dtype=bf16)
    pos = torch.arange(WIN, device=0)[None]

    def masks(config, x):
        kw = {"config": config, "inputs_embeds": x, "attention_mask": None, "past_key_values": None,
              "position_ids": pos}
        return {"full_attention": nh.create_causal_mask(**kw),
                "linear_attention": nh.create_recurrent_attention_mask(**kw)}

    t0 = time.time()
    with torch.no_grad():
        fill(ref.model.embeddings, "backbone.embeddings.")
        for w, ids in enumerate(wins):
            H[0, w] = ref.model.embeddings(torch.tensor([ids], device=0))[0].cpu()
        ref.model.embeddings.to("meta")
        for i, blk in enumerate(ref.model.layers):
            fill(blk, f"backbone.layers.{i}.")
            for w in range(N_WIN):
                x = H[i, w].to(0)[None]
                y = blk(x, attention_mask=masks(ref.config, x).get(blk.block_type), position_ids=pos,
                        past_key_values=None, use_cache=False)
                H[i + 1, w] = y[0].cpu()
            blk.to("meta")
            free()
    ref_s = time.time() - t0
    ref_peak = torch.cuda.max_memory_allocated()
    handles.clear()
    del ref
    free()

    # Phase B: Route 1.
    sampler.phase = "route1_load"
    torch.cuda.reset_peak_memory_stats()
    model, info, load_s = route1.load()
    model.eval()
    sampler.phase = "route1_compare"
    q_dtypes = {}
    quantized = set()
    for mname, mod in model.named_modules():
        if isinstance(mod, bnb.nn.Linear4bit):
            quantized.add(f"{mname}.weight")
        if hasattr(mod, "parametrizations"):
            quantized.update(f"{mname}.{p}" for p in mod.parametrizations)
    for n, t in list(model.named_parameters()) + list(model.named_buffers()):
        n = n.replace(".parametrizations.", ".").replace(".original", "")
        if n not in quantized:
            q_dtypes[n] = str(t.dtype)
    dtype_mismatch = {n: [ref_dtypes.get(n), d] for n, d in q_dtypes.items() if ref_dtypes.get(n) != d}
    q_impl = {"attn": model.config._attn_implementation, "experts": getattr(model.config, "_experts_implementation", None)}
    lm_head_is_bf16_linear = type(model.lm_head) is torch.nn.Linear and model.lm_head.weight.dtype == bf16

    layers = model.model.layers
    types = [b.block_type for b in layers]
    acc_e2 = torch.zeros(n_layers, dtype=torch.float64)
    acc_r2 = torch.zeros(n_layers, dtype=torch.float64)
    acc_cos = torch.zeros(n_layers, dtype=torch.float64)
    iso_e2 = torch.zeros(n_layers, dtype=torch.float64)
    iso_r2 = torch.zeros(n_layers, dtype=torch.float64)
    emb_max_abs_diff = 0.0
    tally = Tally()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for w, ids in enumerate(wins):
            ids_t = torch.tensor([ids], device=0)
            caps = {}
            hooks = [b.register_forward_hook(lambda mod, a, out, i=i: caps.__setitem__(i, out[0]))
                     for i, b in enumerate(layers)]
            hooks.append(model.model.embeddings.register_forward_hook(
                lambda mod, a, out: caps.__setitem__("emb", out[0])))
            last = model.model(input_ids=ids_t, use_cache=False).last_hidden_state
            for h in hooks:
                h.remove()
            emb_max_abs_diff = max(emb_max_abs_diff, (caps["emb"].float() - H[0, w].to(0).float()).abs().max().item())
            for i in range(n_layers):
                r = H[i + 1, w].to(0).float()
                q = caps[i].float()
                acc_e2[i] += (q - r).pow(2).sum().item()
                acc_r2[i] += r.pow(2).sum().item()
                acc_cos[i] += torch.nn.functional.cosine_similarity(q, r, dim=-1).mean().item() / N_WIN
            logits_q = model.lm_head(last)[0]
            logits_r = model.lm_head(model.model.norm_f(H[n_layers, w].to(0)[None]))[0]
            tally.add(logits_r, logits_q, ids_t[0])
            del caps, last, logits_q, logits_r
            for i, blk in enumerate(layers):
                x = H[i, w].to(0)[None]
                y = blk(x, attention_mask=masks(model.config, x).get(blk.block_type), position_ids=pos,
                        past_key_values=None, use_cache=False)
                dq = y[0].float() - x[0].float()
                dr = H[i + 1, w].to(0).float() - x[0].float()
                iso_e2[i] += (dq - dr).pow(2).sum().item()
                iso_r2[i] += dr.pow(2).sum().item()
            free()
    cmp_peak = torch.cuda.max_memory_allocated()

    acc = (acc_e2 / acc_r2).sqrt()
    iso = (iso_e2 / iso_r2).sqrt()

    def by_type(v):
        out = {}
        for t in sorted(set(types)):
            xs = [v[i].item() for i in range(n_layers) if types[i] == t]
            out[t] = {"n": len(xs), "mean": round(sum(xs) / len(xs), 5), "max": round(max(xs), 5)}
        return out

    res = {
        "model": REPO, "revision": REV, "text": meta, "mamba": route1.mamba_kernels(),
        "implementations": {"reference": ref_impl, "route1": q_impl},
        "checks": {"dtype_mismatch_unquantized": dtype_mismatch,
                   "n_unquantized_tensors_compared": len(q_dtypes),
                   "embedding_output_max_abs_diff": emb_max_abs_diff,
                   "lm_head_is_bf16_linear": lm_head_is_bf16_linear,
                   "route1_hook": route1.hook_summary(),
                   "route1_missing_keys": len(info["missing_keys"]),
                   "ckpt_dtypes_non_bf16": {k: v for k, v in ckpt_dtypes.items() if v != "torch.bfloat16"}},
        "reference": {"seconds": round(ref_s, 1), "torch_peak_GiB": gib(ref_peak)},
        "route1": {"load_seconds": round(load_s, 1), "compare_torch_peak_GiB": gib(cmp_peak)},
        "end_to_end": tally.result(),
        "per_layer": {"types": types,
                      "accumulated_rel_err": [round(x, 5) for x in acc.tolist()],
                      "accumulated_mean_token_cos": [round(x, 6) for x in acc_cos.tolist()],
                      "isolated_update_rel_err": [round(x, 5) for x in iso.tolist()]},
        "isolated_by_type": by_type(iso),
        "accumulated_by_type": by_type(acc),
    }
    dump(res)


try:
    run_qwen() if MODE == "qwen" else run_lightning()
finally:
    sampler.stop = True

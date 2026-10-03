"""W70 (docs/next-model-plan.md): can a 70B QLoRA train on the laptop's one 24 GB card with
its 4-bit weights streamed from RAM?

The lab's earlier verdict (gpu-lab 3e41b1e) ruled 70B out because accelerate's CPU offload
keeps offloaded weights in bf16: 2 bytes a parameter, about a 63B ceiling in 61 GiB of RAM.
Here every decoder layer is NF4 (0.44 GiB for a Llama-70B layer), held in pinned RAM and
copied to the GPU only while it computes: 80 layers are 35 GiB of RAM, not 140.

Why not hooks on top of transformers' gradient checkpointing: bitsandbytes' MatMul4Bit keeps
its weight as ctx.tensors = (None, B), an attribute rather than save_for_backward
(bitsandbytes/autograd/_functions.py in 0.50.2), so every streamed layer's GPU copy would
stay alive from its forward to its backward and nothing would be saved. So this file runs
the layers itself, one record at a time:
  forward  : no grad, layer by layer; each layer's input is kept (T x 8192 bf16, 32 MiB at
             2,048 tokens) and the CUDA RNG state before it (LoRA dropout);
  head     : final norm, then chunked cross-entropy over lm_head, backward to its input;
  backward : layer by layer in reverse: reload the layer, restore its RNG state, recompute
             it with grad from the kept input, backward the incoming gradient, unload.
That is gradient checkpointing per layer (forward + recompute + backward, as Q2 runs), and
each layer's autograd graph is freed before the next layer loads. The next layer's weights
are copied on a side stream while the current one computes.

The base: hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4, the 70B already on disk (it
serves pooled). No GPTQ library is in gpu-lab:training, so each GPTQ linear (4-bit, group
128, desc_act, sym) is unpacked to bf16 on the GPU and re-quantized to NF4 with double
quant, as QLoRA loads a bf16 checkpoint. The weights so carry GPTQ's error and NF4's.
Embeddings stay in RAM (the lookup runs on the CPU); lm_head (2 GiB bf16) stays on the GPU.

LoRA: r16, alpha 32, dropout 0.05 on q/k/v/o and up/down (Q2's TARGETS, mapped to Llama);
G6q's optimisation (paged AdamW 8-bit, lr 1e-4, wd 0, cosine with 3% warmup over the 312
steps a full run would take, clip 1.0, 8 records a step, the loss normalised over the step's
assistant tokens, seed 0). Data: G6q's records with Llama 3.1's chat template and the lab's
TOOLS; assistant turns are trained, everything else masked. This is a fit-and-speed probe:
no adapter is saved.

Modes:
  SELFTEST=1  first LAYERS (default 4) layers, one record cut at 512 tokens: (a) plain
              autograd, all resident; (b) this file's loop, all resident; (c) the loop, all
              streamed. Loss and LoRA gradients must match: (b) and (c) bitwise, (a) and (b)
              to bf16 noise. Also the dequant's numbers (scale of W, NF4 error).
  default     all 80 layers: step 1 is the 8 longest records (the memory peak), then
              STEPS-1 steps in G6q's epoch-0 order. RESIDENT=N keeps layers 0..N-1 on the GPU.
Run (laptop): GUARD=hw [SELFTEST=1] [RESIDENT=N] [STEPS=6] \\
  probes/gpurun.sh w70-<name> /probes/w70_stream.py w70-<name>
Output: /out/<label>.json, rewritten after every record.
"""
import gc
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, "/gpulab/training")
from q38 import gib, lightning_messages, tools_for  # noqa: E402

REPO, REV = "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4", "1b0ae7f9d6da8b79f36fdc24912f950ecb2b6e91"
TARGETS = r".*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$|.*\.mlp\.(up_proj|down_proj)$"
HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
label = sys.argv[1]
SELFTEST = os.environ.get("SELFTEST") == "1"
LOSSCHECK = int(os.environ.get("LOSSCHECK") or 0)  # N records through all 80 layers, bf16 and NF4
if SELFTEST:  # cuBLAS's deterministic workspace, set before CUDA starts
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
GUARD = os.environ.get("GUARD", "g5")
DATASET = os.environ.get("DATASET") or "/out/research_dataset_g6q.json"
MAX_LEN = int(os.environ.get("MAX_LEN") or 2048)
RESIDENT = int(os.environ.get("RESIDENT") or 0)
STEPS = int(os.environ.get("STEPS") or 6)
CE_CHUNK = int(os.environ.get("CE_CHUNK", "256"))
EPOCHS, LR, PER_STEP, RANK, SEED, FULL_STEPS = 2, 1e-4, 8, 16, 0, 312
res = {"label": label, "model": REPO, "revision": REV, "selftest": SELFTEST, "guard": GUARD,
       "config": {"max_len": MAX_LEN, "resident": RESIDENT, "steps": STEPS, "lr": LR, "records_per_step": PER_STEP,
                  "rank": RANK, "alpha": 2 * RANK, "dropout": 0.05, "targets": TARGETS, "ce_chunk": CE_CHUNK,
                  "schedule_steps": FULL_STEPS, "quant": "GPTQ int4 -> bf16 -> NF4 double quant, blocksize 64"}}


def dump():
    res["time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(f"/out/{label}.json", "w") as f:
        json.dump(res, f, indent=1)


# ---- data: G6q's records, Llama 3.1's template, assistant turns trained
from huggingface_hub import snapshot_download  # noqa: E402
from transformers import AutoConfig, AutoTokenizer  # noqa: E402

path = snapshot_download(REPO, revision=REV, local_files_only=True)
tok = AutoTokenizer.from_pretrained(path)
header_ids = tok.encode(HEADER, add_special_tokens=False)
ends = {tok.convert_tokens_to_ids("<|eot_id|>"), tok.convert_tokens_to_ids("<|eom_id|>")}


def encode(rec):
    ids = tok.apply_chat_template(lightning_messages(rec["messages"]), tools=tools_for(rec), tokenize=False)
    ids = tok(ids, add_special_tokens=False)["input_ids"]
    labels, i = [-100] * len(ids), 0
    while i < len(ids):
        if ids[i:i + len(header_ids)] == header_ids:
            j = i + len(header_ids)
            while j < len(ids) and ids[j] not in ends:
                labels[j] = ids[j]
                j += 1
            if j < len(ids):
                labels[j] = ids[j]
            i = j + 1
        else:
            i += 1
    return {"ids": ids[:MAX_LEN], "labels": labels[:MAX_LEN], "full_len": len(ids)}


records = json.load(open(DATASET))
data, bad = [], []
for k, r in enumerate(records):
    try:
        data.append(encode(r))
    except Exception as e:  # a record the Llama template cannot render
        bad.append((k, f"{type(e).__name__}: {str(e)[:120]}"))
        data.append(None)
n_items = [0 if d is None else sum(l != -100 for l in d["labels"][1:]) for d in data]
lens = sorted(len(d["ids"]) for d in data if d)
res["data"] = {"records": len(records), "unrenderable": len(bad), "unrenderable_examples": bad[:3],
               "tokens": sum(lens), "assistant_tokens": sum(n_items), "p50": lens[len(lens) // 2],
               "max_full": max(d["full_len"] for d in data if d),
               "truncated": sum(d["full_len"] > MAX_LEN for d in data if d)}
print(json.dumps(res["data"]), flush=True)

# ---- model pieces
import bitsandbytes as bnb  # noqa: E402
import route1  # noqa: E402  (Sampler: NVML peaks, thermal guard)
from peft import LoraConfig, inject_adapter_in_model  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers.models.llama.modeling_llama import (LlamaDecoderLayer, LlamaRMSNorm,  # noqa: E402
                                                      LlamaRotaryEmbedding)
import chunked_ce  # noqa: E402

if GUARD == "hw":
    route1.ABORT_REASONS = ("hw_thermal", "hw_power_brake")
    route1.ABORT_TEMP_C = 90
pynvml, nvh = route1.pynvml, route1.nvh
cfg = AutoConfig.from_pretrained(path, attn_implementation="sdpa")
cfg.quantization_config = None
L = int(os.environ.get("LAYERS") or (4 if SELFTEST else cfg.num_hidden_layers))
dev = torch.device("cuda", 0)
sampler = route1.Sampler(interval=0.25)
sampler.start()
t_start = time.time()
res["platform_profile"] = os.environ.get("PLATFORM_PROFILE", "not passed")
res["nvml_used_before_cuda_GiB"] = gib(pynvml.nvmlDeviceGetMemoryInfo(nvh).used)

wmap = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
handles = {}


def tensor(key):
    f = wmap[key]
    if f not in handles:
        handles[f] = safe_open(os.path.join(path, f), framework="pt", device="cpu")
    return handles[f].get_tensor(key)


SHIFTS = torch.arange(0, 32, 4, dtype=torch.int32, device=dev)


def dequant(prefix, zeros_seen=None):
    """GPTQ (AutoGPTQ v1 layout: zeros stored minus one) -> bf16 weight [out, in], on the GPU.
    Row r of the [in, out] matrix uses group g_idx[r] (desc_act); fp16, GPTQ's scale dtype."""
    qw = tensor(prefix + ".qweight").to(dev)            # int32 [in/8, out]
    qz = tensor(prefix + ".qzeros").to(dev)             # int32 [groups, out/8]
    sc = tensor(prefix + ".scales").to(dev)             # fp16 [groups, out]
    gi = tensor(prefix + ".g_idx").to(dev).long()       # [in]
    w = ((qw.unsqueeze(1) >> SHIFTS[None, :, None]) & 0xF).reshape(-1, qw.shape[1])
    z = ((qz.unsqueeze(2) >> SHIFTS[None, None, :]) & 0xF).reshape(qz.shape[0], -1) + 1
    if zeros_seen is not None:
        zeros_seen.update(z.unique().tolist())
    W = (w - z[gi]).half() * sc[gi]
    del w
    return W.t().contiguous().to(torch.bfloat16)


def linear4(prefix, zeros_seen=None):
    W = dequant(prefix, zeros_seen)
    q, qs = bnb.functional.quantize_4bit(W, blocksize=64, compress_statistics=True, quant_type="nf4")
    lin = bnb.nn.Linear4bit(W.shape[1], W.shape[0], bias=False, compute_dtype=torch.bfloat16,
                            compress_statistics=True, quant_type="nf4", device="meta")
    lin.weight = bnb.nn.Params4bit(data=q, requires_grad=False, quant_state=qs, blocksize=64, compress_statistics=True,
                                   quant_type="nf4", quant_storage=torch.uint8, module=lin, bnb_quantized=True)
    return lin, W


def norm(key):
    n = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dev, torch.bfloat16)
    n.weight.data.copy_(tensor(key))
    n.weight.requires_grad_(False)
    return n


LIN = {"self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"), "mlp": ("gate_proj", "up_proj", "down_proj")}


class Stack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList()


def offload(m):
    """Move one Linear4bit's packed weight and absmax to pinned RAM; keep the copies on m."""
    w, qs = m.weight, m.weight.quant_state
    cw = torch.empty(w.data.shape, dtype=w.data.dtype, pin_memory=True)
    cw.copy_(w.data)
    ca = torch.empty(qs.absmax.shape, dtype=qs.absmax.dtype, pin_memory=True)
    ca.copy_(qs.absmax)
    w.data, qs.absmax = cw, ca
    m._w70_cpu = (cw, ca)


quant_check, zeros_seen = None, set()


def make_layer(i, kind="nf4"):
    """Decoder layer i on the GPU: its linears NF4 (Linear4bit), or kind="bf16", the GPTQ unpack
    itself in plain nn.Linear (LOSSCHECK's reference)."""
    global quant_check
    with torch.device("meta"):
        layer = LlamaDecoderLayer(cfg, layer_idx=i)
    for part, names in LIN.items():
        for n in names:
            if kind == "bf16":
                W = dequant(f"model.layers.{i}.{part}.{n}")
                lin = torch.nn.Linear(W.shape[1], W.shape[0], bias=False, device=dev, dtype=torch.bfloat16)
                lin.weight.data.copy_(W)
                lin.weight.requires_grad_(False)
                setattr(getattr(layer, part), n, lin)
                del W
                continue
            lin, W = linear4(f"model.layers.{i}.{part}.{n}", zeros_seen)
            if quant_check is None:  # layer 0 q_proj: the dequant's scale and NF4's error on it
                back = bnb.functional.dequantize_4bit(lin.weight.data, lin.weight.quant_state).float()
                quant_check = {"W_std": float(W.float().std()), "W_absmax": float(W.float().abs().max()),
                               "nf4_rel_err": float((back - W.float()).norm() / W.float().norm())}
                del back
            setattr(getattr(layer, part), n, lin)
            del W
    layer.input_layernorm = norm(f"model.layers.{i}.input_layernorm.weight")
    layer.post_attention_layernorm = norm(f"model.layers.{i}.post_attention_layernorm.weight")
    assert not [n for n, p in layer.named_parameters() if p.is_meta], i
    return layer


sampler.phase = "load"
t0 = time.time()
embed = tensor("model.embed_tokens.weight").to(torch.bfloat16)  # RAM; lookup on the CPU
final_norm = norm("model.norm.weight")
lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, device=dev, dtype=torch.bfloat16)
lm_head.weight.data.copy_(tensor("model.lm_head.weight") if "model.lm_head.weight" in wmap else tensor("lm_head.weight"))
lm_head.weight.requires_grad_(False)
rotary = LlamaRotaryEmbedding(cfg).to(dev)


def embed_ids(ids):
    return F.embedding(torch.tensor(ids), embed).to(dev, non_blocking=True).unsqueeze(0)


if LOSSCHECK:  # the unpack's proof: every layer built, run on a few records and dropped
    order0 = torch.randperm(len(data), generator=torch.Generator().manual_seed(SEED)).tolist()
    picks = [j for j in order0 if data[j] and n_items[j] > 0][:LOSSCHECK]
    pos = [torch.arange(len(data[j]["ids"]), device=dev).unsqueeze(0) for j in picks]
    hs = {k: [embed_ids(data[j]["ids"]) for j in picks] for k in ("bf16", "nf4")}
    pe = [rotary(hs["bf16"][r], pos[r]) for r in range(len(picks))]
    with torch.no_grad():
        for i in range(cfg.num_hidden_layers):
            for kind in hs:
                layer = make_layer(i, kind)
                hs[kind] = [layer(h, position_ids=pos[r], position_embeddings=pe[r]) for r, h in enumerate(hs[kind])]
                del layer
            if i % 10 == 9:
                torch.cuda.empty_cache()
                print(f"losscheck layer {i + 1} at {time.time() - t0:.0f}s", flush=True)
        out = []
        for r, j in enumerate(picks):
            y_all = torch.tensor([data[j]["ids"]], device=dev)
            y_asst = torch.tensor([data[j]["labels"]], device=dev)
            row = {"record": j, "tokens": len(data[j]["ids"]), "assistant_tokens": n_items[j]}
            for kind, h in hs.items():
                x = final_norm(h[r])
                row[kind] = {"all": round(chunked_ce.chunked_loss(lm_head, x, y_all, CE_CHUNK).item(), 4),
                             "assistant": round(chunked_ce.chunked_loss(lm_head, x, y_asst, CE_CHUNK).item(), 4)}
            out.append(row)
    quant_check["gptq_zeros_seen"] = sorted(zeros_seen)
    res.update({"quant_check": quant_check, "losscheck": out, "wall_s": round(time.time() - t0, 1)})
    sampler.stop = True
    dump()
    print(json.dumps(res["losscheck"], indent=1), flush=True)
    sys.exit(0)

stack = Stack()
for i in range(L):
    layer = make_layer(i)
    if not SELFTEST and i >= RESIDENT:  # the card cannot hold 80 layers, even briefly
        for m in layer.modules():
            if isinstance(m, bnb.nn.Linear4bit):
                offload(m)
    stack.layers.append(layer)
    if i % 10 == 9:
        torch.cuda.empty_cache()
        print(f"layer {i + 1}/{L} at {time.time() - t0:.0f}s, GPU {gib(torch.cuda.memory_allocated())} GiB", flush=True)
quant_check["gptq_zeros_seen"] = sorted(zeros_seen)  # sym=True: every zero should be 8
res["quant_check"] = quant_check
handles.clear()
torch.manual_seed(SEED)
inject_adapter_in_model(LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.05, bias="none",
                                   target_modules=TARGETS), stack)
for n, p in stack.named_parameters():
    if "lora_" in n:  # fp32, as get_peft_model casts them (Q2); inject_adapter_in_model does not
        p.data = p.data.to(dev, torch.float32)
params = [p for n, p in stack.named_parameters() if p.requires_grad]
assert params and all("lora_" in n for n, p in stack.named_parameters() if p.requires_grad)
res["lora"] = {"trainable_params": sum(p.numel() for p in params), "dtype": str(params[0].dtype)}
torch.cuda.synchronize()
res["load_s_quantize"] = round(time.time() - t0, 1)
print(f"quantized {L} layers in {res['load_s_quantize']}s; {quant_check}; {res['lora']}", flush=True)


class Streamer:
    """Layer i's NF4 weights and absmax live in pinned RAM unless i is resident; load(i) copies
    them on a side stream, wait(i) orders the compute stream after the copy, unload(i) drops
    the GPU copies (record_stream keeps the allocator from reusing them early)."""

    def __init__(self, resident):
        self.resident, self.items, self.live, self.bytes = set(resident), [], {}, 0
        self.copy = torch.cuda.Stream()
        for i, layer in enumerate(stack.layers):
            items = []
            for m in layer.modules():
                if isinstance(m, bnb.nn.Linear4bit):
                    if i in self.resident:
                        assert not hasattr(m, "_w70_cpu"), (i, "resident layer was offloaded at load")
                        items.append((m, None, None))
                        continue
                    if not hasattr(m, "_w70_cpu"):
                        offload(m)
                    items.append((m, *m._w70_cpu))
            self.items.append(items)
        gc.collect()
        torch.cuda.empty_cache()
        streamed = [it for it in self.items if it and it[0][1] is not None]
        self.layer_bytes = sum(cw.numel() + ca.numel() for _, cw, ca in streamed[0]) if streamed else 0

    def load(self, i):
        if i < 0 or i >= L or i in self.resident or i in self.live:
            return
        gpu = []
        with torch.cuda.stream(self.copy):
            for m, cw, ca in self.items[i]:
                gw, ga = cw.to(dev, non_blocking=True), ca.to(dev, non_blocking=True)
                m.weight.data, m.weight.quant_state.absmax = gw, ga
                gpu += [gw, ga]
                self.bytes += cw.numel() + ca.numel()
            ev = torch.cuda.Event()
            ev.record(self.copy)
        self.live[i] = (ev, gpu)

    def wait(self, i):
        if i in self.live:
            ev, gpu = self.live[i]
            torch.cuda.current_stream().wait_event(ev)
            for t in gpu:
                t.record_stream(torch.cuda.current_stream())

    def unload(self, i):
        if i in self.live:
            for m, cw, ca in self.items[i]:
                m.weight.data, m.weight.quant_state.absmax = cw, ca
            del self.live[i]


def head_loss(h, labels, scale):
    h = h.detach().requires_grad_(True)
    y = torch.tensor([labels], device=dev)
    loss = chunked_ce.chunked_loss(lm_head, final_norm(h), y, CE_CHUNK) * scale
    loss.backward()
    return loss.item(), h.grad


def record_step(st, ids, labels, scale, timing=None):
    """One record through the layer loop; LoRA grads accumulate. Returns the scaled loss."""
    T = len(ids)
    pos = torch.arange(T, device=dev).unsqueeze(0)
    h = embed_ids(ids)
    pe = rotary(h, pos)
    kept, rng = [], []
    ev = [] if timing is not None else None
    st.load(0)
    with torch.no_grad():
        for i, layer in enumerate(stack.layers):
            st.wait(i)
            st.load(i + 1)
            kept.append(h)
            rng.append(torch.cuda.get_rng_state())
            if ev is not None:
                ev.append(("f", i, torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)))
                ev[-1][2].record()
            h = layer(h, position_ids=pos, position_embeddings=pe)
            if ev is not None:
                ev[-1][3].record()
            st.unload(i)
    st.load(L - 1)
    loss, g = head_loss(h, labels, scale)
    for i in reversed(range(L)):
        st.wait(i)
        st.load(i - 1)
        x = kept[i].detach().requires_grad_(True)
        kept[i] = None
        torch.cuda.set_rng_state(rng[i])
        if ev is not None:
            ev.append(("b", i, torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)))
            ev[-1][2].record()
        with torch.enable_grad():
            y = stack.layers[i](x, position_ids=pos, position_embeddings=pe)
            y.backward(g)
        g = x.grad
        if ev is not None:
            ev[-1][3].record()
        del x, y
        st.unload(i)
    if ev is not None:
        torch.cuda.synchronize()
        for kind, i, a, b in ev:
            key = f"{kind}_{'resident' if i in st.resident else 'streamed'}_ms"
            timing.setdefault(key, []).append(a.elapsed_time(b))
    return loss


def plain_step(ids, labels, scale):
    """Ordinary autograd through all layers (no checkpointing): SELFTEST's reference."""
    T = len(ids)
    pos = torch.arange(T, device=dev).unsqueeze(0)
    h = embed_ids(ids)
    pe = rotary(h, pos)
    for layer in stack.layers:
        h = layer(h, position_ids=pos, position_embeddings=pe)
    y = torch.tensor([labels], device=dev)
    loss = chunked_ce.chunked_loss(lm_head, final_norm(h), y, CE_CHUNK) * scale
    loss.backward()
    return loss.item()


def grads():
    return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]


def zero():
    for p in params:
        p.grad = None


stack.train()
if SELFTEST:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    k = next(j for j, d in enumerate(data) if d and n_items[j] > 50)
    ids, labels = data[k]["ids"][:512], data[k]["labels"][:512]
    n = sum(l != -100 for l in labels[1:])

    def variants(names):
        out = {}
        for name in names:
            st = Streamer(range(L) if name != "loop_streamed" else ())
            zero()
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            torch.cuda.synchronize()
            t = time.time()
            loss = plain_step(ids, labels, 1.0) if name == "plain" else record_step(st, ids, labels, 1.0)
            torch.cuda.synchronize()
            out[name] = {"loss": loss, "grads": grads(), "s": round(time.time() - t, 3)}
            for i in range(L):  # back to all-GPU for the next variant
                for m, cw, ca in st.items[i]:
                    if cw is not None:
                        m.weight.data, m.weight.quant_state.absmax = cw.to(dev), ca.to(dev)
                        del m._w70_cpu
        return out

    def cmp(out, a, b):
        da = [(x - y).abs().max().item() for x, y in zip(out[a]["grads"], out[b]["grads"])]
        ref = max(x.abs().max().item() for x in out[b]["grads"])
        return {"loss_a": out[a]["loss"], "loss_b": out[b]["loss"], "max_abs_grad_diff": max(da),
                "max_abs_grad": ref, "bitwise": max(da) == 0.0 and out[a]["loss"] == out[b]["loss"]}

    names = ("plain", "loop_resident", "loop_resident_again", "loop_streamed")
    with sdpa_kernel(SDPBackend.MATH):  # deterministic kernels: streamed must equal resident bitwise
        torch.use_deterministic_algorithms(True, warn_only=True)
        det = variants(names)
        torch.use_deterministic_algorithms(False)
    fast = variants(names[1:])  # the default kernels, as training runs: their run-to-run floor
    res["selftest_result"] = r = {
        "record": k, "tokens": len(ids), "trained_tokens": n, "layers": L,
        "deterministic": {"resident_vs_again": cmp(det, "loop_resident", "loop_resident_again"),
                          "resident_vs_streamed": cmp(det, "loop_resident", "loop_streamed"),
                          "plain_vs_loop_resident": cmp(det, "plain", "loop_resident")},
        "default_kernels": {"resident_vs_again": cmp(fast, "loop_resident", "loop_resident_again"),
                            "resident_vs_streamed": cmp(fast, "loop_resident", "loop_streamed")},
        "seconds": {"deterministic": {x: det[x]["s"] for x in det}, "default": {x: fast[x]["s"] for x in fast}}}
    d = r["deterministic"]
    r["pass"] = (d["resident_vs_streamed"]["max_abs_grad_diff"] <= d["resident_vs_again"]["max_abs_grad_diff"]
                 and d["resident_vs_streamed"]["loss_a"] == d["resident_vs_streamed"]["loss_b"]
                 and d["plain_vs_loop_resident"]["max_abs_grad_diff"] <= 0.02 * d["plain_vs_loop_resident"]["max_abs_grad"])
    res["torch_peak_GiB"] = gib(torch.cuda.max_memory_allocated())
    sampler.stop = True
    res["phase_peaks"] = sampler.report()
    dump()
    print(json.dumps(res["selftest_result"], indent=1), flush=True)
    sys.exit(0 if r["pass"] else 1)

# ---- the 80-layer run
st = Streamer(range(RESIDENT))
res["streamer"] = {"resident_layers": RESIDENT, "streamed_layers": L - RESIDENT,
                   "layer_GiB": gib(st.layer_bytes), "pinned_GiB": gib(st.layer_bytes * (L - RESIDENT))}
opt = bnb.optim.PagedAdamW8bit(params, lr=LR, weight_decay=0.0)
from transformers import get_cosine_schedule_with_warmup  # noqa: E402
sched = get_cosine_schedule_with_warmup(opt, max(1, round(FULL_STEPS * 0.03)), FULL_STEPS)
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
res["idle"] = {"torch_allocated_GiB": gib(torch.cuda.memory_allocated()),
               "nvml_used_GiB": gib(pynvml.nvmlDeviceGetMemoryInfo(nvh).used)}
res["load_s_total"] = round(time.time() - t_start, 1)
print(f"ready in {res['load_s_total']}s; {res['streamer']}; idle {res['idle']}", flush=True)
valid = [j for j, d in enumerate(data) if d and n_items[j] > 0]
longest = sorted(valid, key=lambda j: -len(data[j]["ids"]))[:PER_STEP]
valid_set = set(valid)
order = [j for j in torch.randperm(len(data), generator=torch.Generator().manual_seed(SEED)).tolist() if j in valid_set]
batches = [longest] + [order[s * PER_STEP:(s + 1) * PER_STEP] for s in range(STEPS - 1)]
res["steps"], oom = [], None
for s, batch in enumerate(batches):
    if sampler.abort:
        break
    n_step = sum(n_items[j] for j in batch)
    phase = f"step{s + 1}"
    sampler.phase = phase
    torch.cuda.reset_peak_memory_stats()
    timing, bytes0, t0 = {}, st.bytes, time.time()
    recs, loss_step = [], 0.0
    try:
        for j in batch:
            tr = time.time()
            loss = record_step(st, data[j]["ids"], data[j]["labels"], n_items[j] / n_step, timing)
            loss_step += loss
            recs.append({"tokens": len(data[j]["ids"]), "seconds": round(time.time() - tr, 2),
                         "loss_unscaled": round(loss * n_step / n_items[j], 4)})
            if sampler.abort:
                break
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
    except torch.OutOfMemoryError as e:
        oom = {"step": s + 1, "message": str(e)[:600], "torch_peak_GiB": gib(torch.cuda.max_memory_allocated())}
        print(f"OOM at step {s + 1}: {str(e)[:300]}", flush=True)
        break
    torch.cuda.synchronize()
    dt = time.time() - t0
    p = sampler.peaks.get(phase, {})
    toks = sum(r["tokens"] for r in recs)
    row = {"step": s + 1, "kind": "8 longest" if s == 0 else "epoch-0 order", "loss": round(loss_step, 4),
           "grad_norm": round(gnorm, 4), "seconds": round(dt, 1), "tokens": toks, "tok_per_s": round(toks / dt, 1),
           "copied_GiB": gib(st.bytes - bytes0), "torch_peak_GiB": gib(torch.cuda.max_memory_allocated()),
           "torch_peak_reserved_GiB": gib(torch.cuda.max_memory_reserved()),
           "nvml_peak_GiB": gib(p.get("nvml_used", 0)), "temp_C": p.get("temp_C"),
           "watts": round(p.get("watts", 0.0), 1), "reasons": {r: v for r, v in p.get("reasons", {}).items() if v},
           "layer_ms_mean": {k: round(sum(v) / len(v), 2) for k, v in timing.items()}, "records": recs}
    res["steps"].append(row)
    print(f"step {s + 1}: loss {row['loss']} gnorm {row['grad_norm']} {row['seconds']}s {row['tok_per_s']} tok/s "
          f"copied {row['copied_GiB']} GiB peak {row['torch_peak_GiB']} nvml {row['nvml_peak_GiB']} "
          f"{row['temp_C']}C {row['layer_ms_mean']}", flush=True)
    res["wall_s"] = round(time.time() - t_start, 1)
    res["phase_peaks"] = sampler.report()
    res["abort"] = sampler.abort
    dump()
res["oom"] = oom
res["abort"] = sampler.abort
normal = [r for r in res["steps"] if r["kind"] == "epoch-0 order"]
if normal:
    tps = sum(r["tokens"] for r in normal) / sum(r["seconds"] for r in normal)
    res["full_run_estimate_h"] = round(2 * res["data"]["tokens"] / tps / 3600, 2)
sampler.stop = True
res["phase_peaks"] = sampler.report()
res["wall_s"] = round(time.time() - t_start, 1)
dump()
print(f"done: {len(res['steps'])} steps, oom={bool(oom)}, abort={sampler.abort}, "
      f"full-run estimate {res.get('full_run_estimate_h')} h", flush=True)
sys.exit(3 if oom else 0)

"""Qwen3.8-27B pieces shared by q1_probe.py (Q1) and q2_train.py (Q2), docs/next-model-plan.md.

Moved out of q1_probe.py unchanged, so that the training run renders, loads and computes its
loss exactly as the probe that passed Q1 did. q1_probe.py's DRY_RUN output after the move is
identical to results/q1-dry.json apart from its timestamp (checked 2026-10-03).

Render: G6q's records with Qwen3.8's template and the lab's TOOLS plus each record's
extra_tools. Thinking-on records render at reasoning_effort EFFORT, with "<think>\\n" masked
(the end of the thinking-on prompt) and "\\n</think>\\n\\n" + content + <|im_end|> trained;
"off" records mask "<think>\\n\\n</think>\\n\\n". A thinking-on record is tokenized in pieces split
after each "<|im_start|>assistant\\n<think>\\n", so that boundary is a token boundary, as the
server's prompt ends there; Render.parity() checks every trained turn's boundary against the
server prompt (g6_train.py's guard).

Load: Qwen3_5ForCausalLM (text only) from the checkpoint's text_config. transformers 5.16.1
maps model.language_model.* onto it (conversion_mapping.py, PrefixChange for qwen3_5_text)
and ignores ^model.visual.* and ^mtp.*, so the vision tower and MTP head are never built.
NF4, double quant, bf16 compute (qlora.py); embeddings and lm_head stay bf16.
caching_allocator_warmup is disabled, as route1.load() does: it only pre-reserves memory.

Loss: install_chunked_ce() replaces Qwen3_5ForCausalLM.forward on the training path (labels
given, grad enabled) with chunked_ce.chunked_loss over the final hidden states; its nemotron_h
install is not used. count_kernel_calls() wraps the DeltaNet and conv1d functions that
modeling_qwen3_5 looks up at call time, to record which path ran.

GPU-side imports stay inside the functions, so the render runs in a container without --gpus.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/gpulab/training")
from tools import TOOLS  # noqa: E402

REPO, REV = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
EFFORT = "low"
TARGETS = (r".*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
           r"|.*\.linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a)$"
           r"|.*\.mlp\.(up_proj|down_proj)$")
HEADER = "<|im_start|>assistant\n"
THINK_OPEN = "<think>\n"
OFF_THINK = "<think>\n\n</think>\n\n"
GiB = 2**30


def gib(n):
    return round(n / GiB, 3)


def lightning_messages(msgs):
    # g6_train.py: the XML tool-call templates want tool-call arguments as objects.
    out = []
    for m in msgs:
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
        out.append(m)
    return out


def tools_for(record):
    return TOOLS + record.get("extra_tools", [])


def kwargs_for(record, effort):
    if record.get("thinking") == "off":
        return {"enable_thinking": False}
    return {"enable_thinking": True, "reasoning_effort": effort}


class Render:
    def __init__(self, tok):
        self.tok = tok
        self.header_ids = tok.encode(HEADER, add_special_tokens=False)
        self.think_open_ids = tok.encode(THINK_OPEN, add_special_tokens=False)
        self.off_think_ids = tok.encode(OFF_THINK, add_special_tokens=False)
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")

    def encode(self, record, effort):
        tok, header_ids, im_end = self.tok, self.header_ids, self.im_end
        off = record.get("thinking") == "off"
        text = tok.apply_chat_template(lightning_messages(record["messages"]), tools=tools_for(record),
                                       tokenize=False, **kwargs_for(record, effort))
        if off:
            pieces = [text]
        else:  # history turns read "<think>\n\n</think>\n\n"; split so "<think>\n" ends a piece
            marker = HEADER + THINK_OPEN
            parts = text.split(marker)
            pieces = [p + marker for p in parts[:-1]] + [parts[-1]]
        ids = []
        for p in pieces:
            if p:
                ids += tok(p, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(ids)
        pre = self.off_think_ids if off else self.think_open_ids
        turns, masked, i = [], 0, 0
        while i < len(ids):
            if ids[i:i + len(header_ids)] == header_ids:
                start = i + len(header_ids)
                if ids[start:start + len(pre)] == pre:
                    start += len(pre)
                    masked += 1
                turns.append((i, start))
                end = start
                while end < len(ids) and ids[end] != im_end:
                    end += 1
                end = min(end + 1, len(ids))
                for j in range(start, end):
                    labels[j] = ids[j]
                i = end
            else:
                i += 1
        return {"ids": ids, "labels": labels, "turns": turns, "masked": masked, "off": off}

    def parity(self, recs, encs, effort):
        """g6_train.py's guard: every trained turn's header-to-first-trained-token span must
        equal the eval server's prompt for that turn (history as the harness sends it, no
        reasoning)."""
        tok, header_ids = self.tok, self.header_ids
        st = {"turns": 0, "boundary_ok": 0, "boundary_bad": 0, "exact": 0, "history_diff_turns": 0,
              "bad_examples": []}
        for rec, d in zip(recs, encs):
            msgs = lightning_messages(rec["messages"])
            ks = [j for j, m in enumerate(msgs) if m["role"] == "assistant"]
            for k, (h, start) in zip(ks, d["turns"]):
                st["turns"] += 1
                srv = tok(tok.apply_chat_template(msgs[:k], tools=tools_for(rec), tokenize=False,
                                                  add_generation_prompt=True, **kwargs_for(rec, effort)),
                          add_special_tokens=False)["input_ids"]
                sh = max(j for j in range(len(srv)) if srv[j:j + len(header_ids)] == header_ids)
                if srv[sh:] == d["ids"][h:start]:
                    st["boundary_ok"] += 1
                else:
                    st["boundary_bad"] += 1
                    if len(st["bad_examples"]) < 3:
                        st["bad_examples"].append({"turn": k, "server": tok.decode(srv[sh:]),
                                                   "train": tok.decode(d["ids"][h:start])})
                if srv == d["ids"][:start]:
                    st["exact"] += 1
                elif srv[:sh] != d["ids"][:h]:
                    st["history_diff_turns"] += 1
        st["pass"] = st["boundary_bad"] == 0 and st["boundary_ok"] > 0
        return st


def counts(encs):
    lens = sorted(len(d["ids"]) for d in encs)
    n = len(lens)

    def pct(q):
        return lens[min(n - 1, int(q * n))]
    return {"records": n, "tokens": sum(lens), "assistant_tokens": sum(sum(l != -100 for l in d["labels"]) for d in encs),
            "p50": pct(0.5), "p90": pct(0.9), "p95": pct(0.95), "max": lens[-1],
            "over_512": sum(x > 512 for x in lens), "over_1024": sum(x > 1024 for x in lens),
            "over_2048": sum(x > 2048 for x in lens),
            "no_trained_tokens": sum(all(l == -100 for l in d["labels"]) for d in encs),
            "off_records": sum(d["off"] for d in encs),
            "turns_with_think_masked": sum(d["masked"] for d in encs)}


def count_kernel_calls(calls):
    """Wrap the functions the DeltaNet forward looks up at call time; calls[name] counts them.
    Returns what each name resolved to and which kernel packages import."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as q35

    def counted(name, fn):
        def wrapper(*a, **k):
            calls[name] = calls.get(name, 0) + 1
            return fn(*a, **k)
        return wrapper

    kpath = {}
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
                 "causal_conv1d_update"):
        fn = getattr(q35, name)
        kpath[name] = {"repr": repr(fn)[:200], "module": getattr(fn, "__module__", None),
                       "qualname": getattr(fn, "__qualname__", None)}
        setattr(q35, name, counted(name, fn))
    for pkg in ("fla", "causal_conv1d", "flash_attn", "kernels"):
        try:
            __import__(pkg)
            kpath[pkg] = "importable"
        except Exception as e:
            kpath[pkg] = type(e).__name__
    return kpath


def install_chunked_ce(chunk, calls):
    import chunked_ce  # its nemotron_h install is not used; only chunked_loss
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as q35
    from transformers.modeling_outputs import CausalLMOutputWithPast

    _orig_forward = q35.Qwen3_5ForCausalLM.forward

    def chunked_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                        inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        if labels is None or not torch.is_grad_enabled():
            return _orig_forward(self, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                 past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                                 use_cache=use_cache, logits_to_keep=logits_to_keep, **kwargs)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                         past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        calls["chunked_ce"] = calls.get("chunked_ce", 0) + 1
        loss = chunked_ce.chunked_loss(self.lm_head, out.last_hidden_state, labels, chunk)
        return CausalLMOutputWithPast(loss=loss, logits=None)

    q35.Qwen3_5ForCausalLM.forward = chunked_forward


def index_prefixes():
    """Key counts of the checkpoint's safetensors index, by top-level prefix."""
    from huggingface_hub import hf_hub_download
    idx_path = os.path.join(os.path.dirname(hf_hub_download(REPO, "config.json", revision=REV)),
                            "model.safetensors.index.json")
    pref = {}
    for k in json.load(open(idx_path))["weight_map"]:
        p = ("model.language_model" if k.startswith("model.language_model.") else
             "model.visual" if k.startswith("model.visual.") else
             "mtp" if k.startswith("mtp.") else k.split(".")[0] if "." in k else k)
        pref[p] = pref.get(p, 0) + 1
    return pref


def load():
    """NF4 load onto cuda:0. Returns (model, loading_info, seconds)."""
    import torch
    import transformers.modeling_utils as mu
    import transformers.models.qwen3_5.modeling_qwen3_5 as q35
    from transformers import AutoConfig, BitsAndBytesConfig

    tcfg = AutoConfig.from_pretrained(REPO, revision=REV).text_config
    mu.caching_allocator_warmup = lambda *a, **k: None
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
    t0 = time.time()
    model, info = q35.Qwen3_5ForCausalLM.from_pretrained(
        REPO, revision=REV, config=tcfg, quantization_config=bnb_cfg, dtype=torch.bfloat16,
        device_map={"": 0}, output_loading_info=True)
    return model, info, time.time() - t0


def inventory(model, info):
    """What the load built: keys, module classes, devices, resident bytes by class."""
    import bitsandbytes as bnb
    import torch

    lin4 = [n for n, m in model.named_modules() if isinstance(m, bnb.nn.Linear4bit)]
    lin16 = [n for n, m in model.named_modules() if type(m) is torch.nn.Linear]
    packed = sum(p.numel() * p.element_size() for p in model.parameters() if p.dtype == torch.uint8)
    rest = {}
    for n, p in model.named_parameters():
        if p.dtype != torch.uint8:
            rest[str(p.dtype)] = rest.get(str(p.dtype), 0) + p.numel() * p.element_size()
    return {"class": type(model).__name__,
            "missing_keys": sorted(info["missing_keys"]), "unexpected_keys": sorted(info["unexpected_keys"])[:20],
            "n_unexpected": len(info["unexpected_keys"]),
            "mismatched_keys": [str(x) for x in info.get("mismatched_keys", [])][:20],
            "n_linear4bit": len(lin4), "bf16_linear_modules": lin16,
            "visual_modules": sum("visual" in n for n, _ in model.named_modules()),
            "mtp_modules": sum(n.startswith("mtp") for n, _ in model.named_modules()),
            "param_devices": sorted({str(p.device) for p in model.parameters()}),
            "attn_implementation": model.config._attn_implementation,
            "resident": {"nf4_packed_GiB": gib(packed),
                         "other_params_GiB": {k: gib(v) for k, v in rest.items()},
                         "allocated_minus_params_GiB": gib(torch.cuda.memory_allocated() - packed
                                                           - sum(rest.values()))}}

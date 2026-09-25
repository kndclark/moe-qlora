"""Chunked cross-entropy for the nemotron_h causal LM: lm_head and the loss run a chunk of
positions at a time, each chunk under activation checkpointing.

Why (David approved trying it 2026-09-25, to fit G5 at seq 2048 on the laptop):
transformers 5.16.1 NemotronHForCausalLM.forward (modeling_nemotron_h.py:1170) runs
lm_head over every position and upcasts to fp32, then ForCausalLMLoss (loss_utils.py)
runs F.cross_entropy on that. At seq 2048 and vocab 131072 (ARITHMETIC): bf16 logits
0.5 GiB, fp32 logits 1.0 GiB, cross_entropy's saved log-softmax 1.0 GiB, and fp32
gradients of the same size in its backward; the returned output also keeps the fp32
logits alive through the backward. Here only the final hidden states are kept
(2048 x 2688 bf16 = 11 MiB); a chunk's logits exist only while that chunk runs, once in
the forward and once more in the backward (recompute).

The idea is NVIDIA NeMo AutoModel's _ChunkedCrossEntropySum (chunked_ce.py at
d9334f362fdb3cc71358155961c7b3b402259d75), which chunks the fp32 upcast but saves the full
bf16 logits. This also chunks the projection, so it covers a LoRA on lm_head as well.
Same math as ForCausalLMLoss: labels shifted by one, ignore_index -100, the sum over valid
positions divided by their count (or by num_items_in_batch when given), which is the mean
that fixed_cross_entropy takes. Cost: lm_head runs twice per chunk.

Only the training path (labels given, grad enabled) changes, and its output has
logits=None. Everything else goes to the original forward.
"""
import torch
import torch.nn.functional as F
import transformers.models.nemotron_h.modeling_nemotron_h as nh
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutputWithPast

IGNORE = -100
_orig_forward = nh.NemotronHForCausalLM.forward
state = {"chunk": 256, "calls": 0, "installed": False}


def _chunk_loss_sum(lm_head, h, y):
    logits = lm_head(h).float()
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1), ignore_index=IGNORE,
                           reduction="sum")


def chunked_loss(lm_head, hidden, labels, chunk, num_items_in_batch=None):
    labels = F.pad(labels, (0, 1), value=IGNORE)[..., 1:]
    total = hidden.new_zeros((), dtype=torch.float32)
    for s in range(0, hidden.shape[1], chunk):
        h, y = hidden[:, s:s + chunk], labels[:, s:s + chunk]
        if torch.is_grad_enabled():
            total = total + checkpoint(_chunk_loss_sum, lm_head, h, y, use_reentrant=False)
        else:
            total = total + _chunk_loss_sum(lm_head, h, y)
    n = num_items_in_batch if num_items_in_batch is not None else (labels != IGNORE).sum()
    return total / n


def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
            inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
    if labels is None or not torch.is_grad_enabled():
        return _orig_forward(self, input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                             logits_to_keep=logits_to_keep, **kwargs)
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                         past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                         **kwargs)
    state["calls"] += 1
    loss = chunked_loss(self.lm_head, outputs[0], labels, state["chunk"], kwargs.get("num_items_in_batch"))
    return CausalLMOutputWithPast(loss=loss, logits=None, past_key_values=outputs.past_key_values,
                                  hidden_states=outputs.hidden_states, attentions=outputs.attentions)


def install(chunk=256):
    state["chunk"] = int(chunk)
    nh.NemotronHForCausalLM.forward = forward
    state["installed"] = True


def uninstall():
    nh.NemotronHForCausalLM.forward = _orig_forward
    state["installed"] = False


def report():
    return {"chunk": state["chunk"], "calls": state["calls"],
            "installed": nh.NemotronHForCausalLM.forward is forward}

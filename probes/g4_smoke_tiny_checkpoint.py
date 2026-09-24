"""G4 smoke fixture: a tiny random nemotron_h saved in Lightning's own layout.

save_pretrained reverts transformers' expert fusion, so the file holds
backbone.layers.N.mixer.experts.K.{up,down}_proj.weight, the same per-expert
keys as Lightning's shards. Loading it back runs the real MergeModulelist path
and the g4_route1_load.py hook. vocab_size matches the Qwen3-8B tokenizer used
as the smoke-test stand-in.
  docker run --rm --init -v $PWD/probes:/probes:ro -v TINY_DIR:/tiny \
    --user $(id -u):$(id -g) -e HF_HOME=/tmp/hf --entrypoint python3 \
    gpu-lab:training /probes/g4_smoke_tiny_checkpoint.py
then run g4_route1_load.py with -v TINY_DIR:/tiny:ro -e G4_MODEL_DIR=/tiny
-e G4_TOKENIZER=/hf/hub/models--Qwen--Qwen3-8B/snapshots/<rev>.
"""
import torch, json
from safetensors import safe_open
from transformers import NemotronHConfig, NemotronHForCausalLM
torch.manual_seed(0)
cfg = NemotronHConfig(
    vocab_size=151936, hidden_size=256,
    layers_block_type=["mamba", "moe", "attention", "moe"],
    num_attention_heads=4, num_key_value_heads=2, head_dim=64, intermediate_size=512,
    mamba_num_heads=8, mamba_head_dim=32, ssm_state_size=16, n_groups=2, chunk_size=32,
    n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=128,
    moe_shared_expert_intermediate_size=256, n_group=1, topk_group=1,
    routed_scaling_factor=2.5, mlp_hidden_act="relu2", num_nextn_predict_layers=0, use_cache=False)
m = NemotronHForCausalLM(cfg).to(torch.bfloat16)
m.save_pretrained("/tiny")
with safe_open("/tiny/model.safetensors", "pt") as f:
    keys = list(f.keys())
print(len(keys), [k for k in keys if "expert" in k][:4])

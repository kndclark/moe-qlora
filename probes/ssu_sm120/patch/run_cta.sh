#!/bin/bash
# Like ../gpu/run.sh, plus the three patched SSU headers and a fresh JIT cache.
N=ssu-cta-$1; shift
H=$(cd "$(dirname "$0")/.." && pwd); C=${SSU_CACHE:-$HOME/.cache/ssu_sm120}; mkdir -p $C/home
exec docker run --rm --name "$N" --gpus all --network none --ipc=host --user 1000:1000 \
  -v /srv/model-cache:/hf:ro -v $H/gpu:/work -v $C:/cache -w /work \
  -v $H/patch/include/kernel_selective_state_update_stp.cuh:/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/mamba/kernel_selective_state_update_stp.cuh:ro \
  -v $H/patch/include/kernel_selective_state_update_mtp_vertical.cuh:/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/mamba/kernel_selective_state_update_mtp_vertical.cuh:ro \
  -v $H/patch/include/kernel_selective_state_update_mtp_horizontal.cuh:/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/mamba/kernel_selective_state_update_mtp_horizontal.cuh:ro \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e HOME=/cache/home \
  -e FLASHINFER_WORKSPACE_BASE=/cache/fiws_cta -e TRITON_CACHE_DIR=/cache/triton \
  -e PYTHONPATH=/work --entrypoint python3 vllm/vllm-openai:v0.29.0 "$@"

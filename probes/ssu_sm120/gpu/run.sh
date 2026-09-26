#!/bin/bash
# Usage: run.sh <name-suffix> <script.py> [args...]   -- throwaway, no network
W=$(cd "$(dirname "$0")" && pwd); C=${SSU_CACHE:-$HOME/.cache/ssu_sm120}; mkdir -p $C/home
N=ssu-probe-$1; shift
exec docker run --rm --name "$N" --gpus all --network none --ipc=host --user 1000:1000 \
  -v /srv/model-cache:/hf:ro -v $W:/work -v $C:/cache -w /work \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e HOME=/cache/home \
  -e FLASHINFER_WORKSPACE_BASE=/cache/fiws -e TRITON_CACHE_DIR=/cache/triton \
  -e PYTHONPATH=/work ${EXTRA_ENV} \
  --entrypoint python3 vllm/vllm-openai:v0.29.0 "$@"

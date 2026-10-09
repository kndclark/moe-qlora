#!/bin/bash
# Usage: sweep.sh <dir> <node>   (runs on that node; writes <node>-h2d-sweep.jsonl)
D=$1; NODE=$2
timeout 600 docker run --rm --name h2d-sweep --pull never --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
  --entrypoint python3 -v "$D:/w" vllm/vllm-openai:v0.29.0 -u /w/${3:-h2d_sweep}.py > "$D/$NODE-${3:-h2d_sweep}.jsonl" 2> "$D/$NODE-${3:-h2d_sweep}.err"
echo "$NODE sweep rc=$?"

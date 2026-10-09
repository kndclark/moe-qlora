#!/bin/bash
# Re-run Row L's copy probe unchanged (read-only mount of probes/kstage)
K=/home/david/moe-qlora/.claude/worktrees/next-round/probes/kstage
timeout 600 docker run --rm --name h2d-rowl --pull never --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 -v "$K:/k:ro" vllm/vllm-openai:v0.29.0 -u /k/copy_bench.py 1024 2>&1 | grep -v Warning | tail -25

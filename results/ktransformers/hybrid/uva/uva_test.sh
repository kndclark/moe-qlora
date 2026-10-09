#!/bin/bash
# uva_test.py on this node: $1 = dir holding uva_test.py, $2 = kstage dir.
H=${1:-/home/david/handoff-0948db4b-files/hybrid/uva}
K=${2:-/home/david/moe-qlora/.claude/worktrees/next-round/probes/kstage}
timeout 600 docker run --rm --name kuva-test --pull never --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 -v "$H:/w" -v "$K:/k:ro" vllm/vllm-openai:v0.29.0 \
  -u /w/uva_test.py 2>&1 | grep -v Warning | tail -12
echo "rc=${PIPESTATUS[0]}"

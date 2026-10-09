#!/bin/bash
# thp_test.py in the vLLM image, kstage mounted read-only.
H=/home/david/handoff-0948db4b-files/hybrid/thp
K=/home/david/moe-qlora/.claude/worktrees/next-round/probes/kstage
timeout 600 docker run --rm --name kthp-test --pull never --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 -e KSTAGE_THP=1 -e KSTAGE_COPY_LIVE=82 --entrypoint python3 \
  -v "$H:/w" -v "$K:/k:ro" vllm/vllm-openai:v0.29.0 -u /w/thp_test.py 2>&1 | grep -v Warning | tail -15
echo "rc=${PIPESTATUS[0]}"

#!/bin/bash
# Run h2d_vmm.py on the laptop with probes/kstage mounted read-only at /k
H=/home/david/handoff-0948db4b-files/hybrid
K=/home/david/moe-qlora/.claude/worktrees/next-round/probes/kstage
timeout 600 docker run --rm --name h2d-vmm --pull never --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 -v "$H:/w" -v "$K:/k:ro" vllm/vllm-openai:v0.29.0 \
  -u /w/h2d_vmm.py > "$H/laptop-h2d_vmm.jsonl" 2> "$H/laptop-h2d_vmm.err"
echo "rc=$?"
cat "$H/laptop-h2d_vmm.jsonl"
grep -v Warning "$H/laptop-h2d_vmm.err" | tail -5

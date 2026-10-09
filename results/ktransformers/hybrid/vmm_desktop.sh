#!/bin/bash
# Desktop twin of vmm.sh: vllm_kstage.py copied to ./k (read-only mount)
H=$HOME/kt-relu2-results/hybrid
timeout 900 docker run --rm --name h2d-vmm --pull never --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 -v "$H:/w" -v "$H/k:/k:ro" vllm/vllm-openai:v0.29.0 \
  -u /w/h2d_vmm.py > "$H/desktop-h2d_vmm.jsonl" 2> "$H/desktop-h2d_vmm.err"
echo "vmm rc=$?"

#!/bin/bash
# h2d_thp2.py on the laptop: registered 4k vs THP, then torch's pinned allocator plain and with the
# malloc + cudaHostRegister config (with and without glibc's THP tunable).
H=${1:-/home/david/.claude/jobs/0948db4b/tmp/hybrid}
K=${2:-/home/david/moe-qlora/.claude/worktrees/next-round/probes/kstage}; N=${3:-laptop}
OUT=$H/$N-h2d_thp2.jsonl
: > "$OUT"
run() {
  timeout 600 docker run --rm --name h2d-thp2 --pull never --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -e PYTHONDONTWRITEBYTECODE=1 "$@" --entrypoint python3 -v "$H:/w" -v "$K:/k:ro" vllm/vllm-openai:v0.29.0 \
    -u /w/h2d_thp2.py "${ARMS[@]}" >> "$OUT" 2>> "$H/$N-h2d_thp2.err"
  echo "rc=$?"
}
ARMS=(reg_4k reg_thp torch_pin); run
ARMS=(torch_pin); run -e PYTORCH_CUDA_ALLOC_CONF=pinned_use_cuda_host_register:True
ARMS=(torch_pin); run -e PYTORCH_CUDA_ALLOC_CONF=pinned_use_cuda_host_register:True -e GLIBC_TUNABLES=glibc.malloc.hugetlb=1
cat "$OUT"
grep -v Warning "$H/$N-h2d_thp2.err" | tail -5

#!/bin/bash
# Serve-level A/B of vLLM's Mamba SSU backend for Lightning NVFP4 on the laptop.
# Usage: RUN=r2 serve_ab.sh LABEL [--docker ARGS...] [--vllm ARGS...]   (run_all.sh sets RUN)
#   LABEL      output prefix, e.g. triton or flashinfer-fix1
#   --docker   extra `docker run` args (bind-mounted patch, -e VAR=...)
#   --vllm     extra vLLM args (e.g. --mamba-backend flashinfer)
# Same flags as G7a's working serve otherwise (moe-qlora docs/plan.md, G7a).
# Measures: readiness, bench.py decode rate at c=1 and c=16, and the v1 eval
# (thinking off) for comparison with results/research-eval-v1-lightning-nothink.json.
set -u
label=$1; shift
dargs=(); vargs=(); mode=
for a in "$@"; do
  case "$a" in --docker) mode=d ;; --vllm) mode=v ;;
    *) [ "$mode" = d ] && dargs+=("$a") || vargs+=("$a") ;; esac
done
out=$(cd "$(dirname "$0")/../../.." && pwd)/results/ssu_sm120/serve/${RUN:?set RUN, e.g. r2}/$label
mkdir -p "$out"
name=ssu-ab-$label
B=http://127.0.0.1:8301

restore() {
  docker stop "$name" >/dev/null 2>&1
  docker logs "$name" > "$out/serve.log" 2>&1 || true
  docker rm "$name" >/dev/null 2>&1
  echo performance | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null
  echo "profile: $(cat /sys/firmware/acpi/platform_profile)"
}
trap restore EXIT

echo max-power | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null
docker run -d --name "$name" --gpus all --ipc=host -p 127.0.0.1:8301:8000 \
  -v /srv/model-cache:/hf:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 "${dargs[@]}" \
  vllm/vllm-openai:v0.29.0 \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --revision bee7596271d1495f6992ae224aefde4410e816b8 --served-model-name lightning-nvfp4 \
  --kv-cache-dtype fp8 --mamba-cache-mode align --moe-backend marlin \
  --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.85 --enforce-eager \
  "${vargs[@]}" >/dev/null || exit 1

t0=$(date +%s)
until curl -sf $B/health >/dev/null; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
    echo "$label: container exited before ready ($(( $(date +%s)-t0 ))s)"; exit 2; fi
  [ $(( $(date +%s)-t0 )) -gt 600 ] && { echo "$label: not ready in 600s"; exit 3; }
  sleep 5
done
echo "$label: ready in $(( $(date +%s)-t0 ))s"
docker logs "$name" 2>&1 | grep -E "SSU|Mamba SSU|mamba_ssm_cache_dtype|KV cache|Available KV" | sed 's/^/  /'
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/  card: /'

cd ~/gpu-lab
python3 bench/bench.py --base $B --model lightning-nvfp4 --max-tokens 256 \
  --concurrency 1 --repeats 3 --label "$label-c1" --json-out "$out/bench-c1.json" > "$out/bench-c1.log" 2>&1
echo "$label: bench c1 exit=$?"
python3 bench/bench.py --base $B --model lightning-nvfp4 --max-tokens 256 \
  --concurrency 16 --repeats 8 --label "$label-c16" --json-out "$out/bench-c16.json" > "$out/bench-c16.log" 2>&1
echo "$label: bench c16 exit=$?"
cd ~/moe-qlora
t1=$(date +%s)
python3 probes/g7a_eval.py --base $B --out "$out/research-eval-v1-nothink.json" \
  --label "v1-lightning-nothink-$label" --set v1 --thinking off > "$out/research-eval-v1-nothink.log" 2>&1
echo "$label: eval v1 exit=$? $(( $(date +%s)-t1 ))s"

#!/bin/bash
# The agent loads of kv_levers.sh (AGENT8L / AGENT16L: 8 or 16 agents growing to 127k tokens) on
# the pool: Lightning NVFP4 with the G6q adapter across both cards (pipeline parallel 2 through
# gpu-lab's `lab pool up`), CUDA graphs, G7c2's 16 sequences at 0.85. Prefill chunks are
# POOL_MAX_BATCHED_TOKENS, 4096 here to match the laptop arms' b4k (the lab's default is 512).
# G7c2's pool held ~2.1M tokens of KV, enough for 16 agents at 127k without host RAM, which on
# one card takes host KV or K6. Ends with `lab pool down` and `lab up` on the desktop, as
# g7c_pool.sh does. The pool binds to the direct link, which is llm's ssh address; that address
# is written into nothing kept here (outputs are redacted on exit: the repo is public).
# PA_KV=bfloat16 trades KV room (4.8M fp8 tokens, ~2x what 16 agents need) for the faster bf16
# decode p41-p42 found on the laptop.
# usage: [POOL_MAX_BATCHED_TOKENS=n] [PA_OUT=dir] [PA_NOLORA=1] [PA_KV=dtype] pool_agents.sh LOAD...   (a8l, a16l)
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
out=$here/results/${PA_OUT:-pool-agents}
mkdir -p "$out"
LAB=/home/david/gpu-lab/bin/lab
M=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
addr=$(ssh -G llm | awk '/^hostname /{print $2}')
B=http://$addr:8200
model=g6q
lora="--enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules g6q=/hf/adapters/lightning-g6q"
lora="$lora --lora-target-modules q_proj k_proj v_proj o_proj in_proj up_proj down_proj"
[ "${PA_NOLORA:-0}" = 1 ] && { lora=""; model=$M; }
finish() {
  ssh llm "sudo docker exec ray-head cat /tmp/vllm-pool.log" > "$out/vllm-pool.log" 2>&1 < /dev/null
  echo "  max running/waiting: $(grep -oE "Running: [0-9]+ reqs, Waiting: [0-9]+ reqs" "$out/vllm-pool.log" | sort -t' ' -k2,2n -k5,5n | tail -1)"
  "$LAB" pool down > "$out/pool-down.log" 2>&1; echo "pool down exit $?"
  ssh llm "$LAB up" > "$out/desktop-lab-up.log" 2>&1 < /dev/null
  echo "desktop lab up exit $?; llama-swap $(ssh llm systemctl is-active llama-swap < /dev/null)"
  local net=${addr%.*}   # the direct link's /24, both ends
  sed -i -E "s/${net//./\\.}\.[0-9]+/LINK/g" "$out"/*
}
trap finish EXIT
POOL_MODEL=$M POOL_MAXLEN=131072 POOL_WAIT_SECS=900 POOL_GPU_UTIL=${POOL_GPU_UTIL:-0.85} \
POOL_MAX_BATCHED_TOKENS=${POOL_MAX_BATCHED_TOKENS:-4096} \
POOL_EXTRA_FLAGS="--revision bee7596271d1495f6992ae224aefde4410e816b8 --kv-cache-dtype ${PA_KV:-fp8} --mamba-cache-mode align --moe-backend marlin --linear-backend marlin --max-num-seqs 16 --enable-prompt-tokens-details${lora:+ $lora}" \
  "$LAB" pool up > "$out/pool-up.log" 2>&1
rc=$?
echo "== pool up exit $rc $(date -u +%H:%M:%SZ) chunks ${POOL_MAX_BATCHED_TOKENS:-4096} model $model kv ${PA_KV:-fp8}"
if [ $rc -ne 0 ]; then
  ssh llm "sudo docker exec ray-head cat /tmp/vllm-pool.log" < /dev/null 2>/dev/null | grep -E "Error|error" | tail -4 | cut -c1-240
  exit 1
fi
ssh llm "sudo docker exec ray-head grep -E 'KV cache size|Maximum concurrency|Available KV|Model loading took' /tmp/vllm-pool.log" < /dev/null | sed 's/^.*INFO/INFO/' | cut -c1-200
for load in "$@"; do
  case $load in
    a8l) w="--agents 8 --start 8192 --chunk 4096 --gen 256 --target 126976" ;;
    a16l) w="--agents 16 --start 8192 --chunk 4096 --gen 256 --target 126976" ;;
    *) echo "unknown load $load"; continue ;;
  esac
  t=$(date +%s)
  # shellcheck disable=SC2086  # w is a word list
  timeout 2400 python3 "$here/probes/offload_bench.py" --base "$B" --model "$model" --out "$out/bench-$load.json" agent $w \
    > "$out/bench-$load.log" 2>&1
  echo "  $load: exit $?, $(( $(date +%s)-t ))s"; grep -E "^  agent" "$out/bench-$load.log"
done

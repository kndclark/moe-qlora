#!/usr/bin/env bash
# Row N, first candidate: vLLM PR #37190's GPU expert cache (--moe-expert-cache-size) on Lightning.
# The PR caches BF16 and FP8 experts only, not NVFP4, so these arms serve the BF16 checkpoint quantized
# to FP8 at load (--quantization fp8) on the Triton MoE kernel, the one FP8 path the cache accepts.
# In FP8 the 23 MoE layers' 2,944 experts take 27.4 GiB, more than the card: the cache keeps N per
# layer on the GPU and copies the rest from pinned RAM on a miss. The PR refuses LoRA, so these arms
# run without the G6q adapter; p35's baselines ran with it.
# PyTorch rounds each pinned allocation up to a power of two, so a layer's 609 MiB expert blocks
# took 1 GiB each: c64-smoke pinned +49,972 MiB. pinned_max_round_threshold_mb:64 stops that.
# Image: vllm-openai:v0.29.0-pr37190 (probes/pr37190/build.sh). Laptop only.
# Usage: probes/pr37190.sh ARM...   ARM = c<N>[-tok][-kvoff8|-kvoff16]-{smoke|pref|a8l|a16l}
#   c<N>   N experts per layer on the GPU; -tok splits an overfull batch by rows, not by experts
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
J=$here/results/pr37190; mkdir -p "$J"
B=http://127.0.0.1:8313
name=pr37190-tm
M=lightning-bf16-fp8
trap 'docker rm -f "$name" >/dev/null 2>&1' EXIT

phase() {  # $1 arm; the rest are vLLM flags
  local tag=$1; shift
  echo "== $tag $(date -u +%H:%M:%SZ)"
  if sudo -n true 2>/dev/null; then   # pinned RAM wants free 2 MiB blocks (see kv_levers.sh phase)
    sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'
  fi
  local ram0=$(free -m | awk '/^Mem:/{print $3}')
  docker run -d --name "$name" --gpus all --ipc=host -p 127.0.0.1:8313:8000 \
    -v /srv/model-cache:/hf:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e DO_NOT_TRACK=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=pinned_max_round_threshold_mb:64 --pull never vllm-openai:v0.29.0-pr37190 \
    --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
    --revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9 --served-model-name $M \
    --quantization fp8 --moe-backend triton \
    --kv-cache-dtype fp8 --mamba-cache-mode align --max-model-len 131072 --max-num-seqs 16 \
    --gpu-memory-utilization 0.92 --max-num-batched-tokens 4096 "$@" >/dev/null || return 1
  local t0=$(date +%s)
  until curl -sf $B/health >/dev/null; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
      echo "container exited before ready ($(( $(date +%s)-t0 ))s)"; docker logs "$name" > "$J/serve-$tag.log" 2>&1
      grep -E "Error|error" "$J/serve-$tag.log" | tail -4 | cut -c1-300
      docker rm -f "$name" >/dev/null 2>&1; return 2; fi
    [ $(( $(date +%s)-t0 )) -gt 1800 ] && { echo "not ready in 1800s"; docker logs "$name" > "$J/serve-$tag.log" 2>&1; docker rm -f "$name" >/dev/null 2>&1; return 3; }
    sleep 5
  done
  echo "ready in $(( $(date +%s)-t0 ))s; $(docker logs "$name" 2>&1 | grep -oE "Model loading took [0-9.]+ GiB|Available KV cache memory: [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens" | tr '\n' ';')"
  echo "  mem: $(docker logs "$name" 2>&1 | grep -oE "Actual usage is [0-9.]+ GiB for consumed memory \(weights \+ non-torch\), [0-9.]+ GiB for peak activation, and [0-9.]+ GiB for CUDAGraph|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" | tr '\n' ';')"
  echo "  cache: $(docker logs "$name" 2>&1 | grep -c "Expert LRU cache enabled") layers, $(docker logs "$name" 2>&1 | grep -oE "Expert LRU cache enabled for [^:]+: [0-9]+/[0-9]+" | head -1)"
  echo "  backend: $(docker logs "$name" 2>&1 | grep -oE "Using [A-Za-z_0-9]+ (Fp8 MoE|MoE) backend[^.]*|downgrading cudagraph_mode [A-Za-z_.]+ -> [A-Z]+" | sort -u | tr '\n' ';')"
  echo "  host RAM used: +$(( $(free -m | awk '/^Mem:/{print $3}') - ram0 )) MiB, $(free -m | awk '/^Mem:/{print $7}') MiB available"
  local p
  for p in "2+2=" "The capital of France is"; do
    echo "  smoke $p $(curl -s $B/v1/completions -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"prompt\":\"$p\",\"max_tokens\":8,\"temperature\":0}" | python3 -c 'import json,sys; print(repr(json.load(sys.stdin)["choices"][0]["text"]))' 2>&1)"
  done
  if [ -n "${WORK:-}" ]; then
    local t=$(date +%s)
    # shellcheck disable=SC2086  # WORK and BARGS are word lists
    # the client gets 40 minutes; a stall ends the bench, never the container (its log is kept)
    timeout 2400 python3 "$here/probes/offload_bench.py" --base $B --model $M --out "$J/bench-$tag.json" $WORK ${BARGS:-} \
      > "$J/bench-$tag.log" 2>&1
    echo "  bench: exit $?, $(( $(date +%s)-t ))s"; grep -E "^  (decode|prefill|agent)" "$J/bench-$tag.log"
  fi
  docker logs "$name" > "$J/serve-$tag.log" 2>&1
  echo "  max running/waiting: $(grep -oE "Running: [0-9]+ reqs, Waiting: [0-9]+ reqs" "$J/serve-$tag.log" | sort -t' ' -k2,2n -k5,5n | tail -1)"
  docker rm -f "$name" >/dev/null 2>&1
}

run() {
  local a=$1 o=() n=${1#c}; n=${n%%-*}
  [[ $n =~ ^[0-9]+$ ]] || { echo "bad arm $a"; return 9; }
  o=(--moe-expert-cache-size "$n" --moe-expert-cache-split expert)
  case $a in *-tok-*) o[3]=token ;; esac
  case $a in *-kvoff8-*) o+=(--kv-offloading-size 8 --kv-offloading-backend native) ;;
             *-kvoff16-*) o+=(--kv-offloading-size 16 --kv-offloading-backend native) ;; esac
  case $a in
    *-smoke) phase "$a" "${o[@]}" ;;
    *-pref) WORK="decode prefill" BARGS="--conc 1,16 --lens 8192,32768,98304" phase "$a" "${o[@]}" ;;
    *-a8l) WORK=agent BARGS="--agents 8 --start 8192 --chunk 4096 --gen 256 --target 126976" phase "$a" "${o[@]}" ;;
    *-a16l) WORK=agent BARGS="--agents 16 --start 8192 --chunk 4096 --gen 256 --target 126976" phase "$a" "${o[@]}" ;;
    *) echo "bad arm $a"; return 9 ;;
  esac
}
for a in "$@"; do run "$a"; done

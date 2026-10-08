#!/bin/bash
# kv_levers.sh's agent arms (p31-p40) on the desktop's RTX 3090 (sm_86). The desktop moves host
# RAM to the GPU at 12.2 GB/s against the laptop's 51.8 (docs/laptop-memory-levers.md, row L),
# and host KV and the K6 expert cache both ride that link, so their walls should move here.
# Runs on the desktop from a copied probes/ tree (~/moe-qlora is not a checkout there):
#   scp -r probes results/kv-levers/profiles llm:~/moe-qlora-agents/   (profiles/ beside probes/)
#   ssh llm 'cd ~/moe-qlora-agents && nohup bash probes/desktop_agents.sh ARM... > results/d41.out 2>&1 &'
# Arms are named as in kv_levers.sh: d41-base-kvoff8-b4k-a8l is the laptop's p33/p40 control.
# Lightning on sm_86 needs --linear-backend marlin. Only this script's own container is touched:
# grafana, prometheus and litellm keep running, and there is no platform profile to set.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here" || exit 9
J=$here/results
mkdir -p "$J"
adapter=/srv/model-cache/adapters/lightning-g6q
B=http://127.0.0.1:8323
name=agents-3090
TM=(--lora-target-modules q_proj k_proj v_proj o_proj in_proj up_proj down_proj)
LORA=(--enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules "g6q=/adapter")
AGENT8L="--agents 8 --start 8192 --chunk 4096 --gen 256 --target 126976"
AGENT16L="--agents 16 --start 8192 --chunk 4096 --gen 256 --target 126976"
# K6 as in kv_levers.sh p31-p40; COPY_LIVE=82 is one copy per SM, and the 3090 has 82 SMs too
KC6="KSTAGE=cache KSTAGE_COLD_GB=4 KSTAGE_PROFILE=/prof/p7-all.json KSTAGE_STATS=1 KSTAGE_SLOTS=all"
KC6="$KC6 KSTAGE_DMA_M=64 KSTAGE_DMA_BUF=1 KSTAGE_COPY_LIVE=82 KSTAGE_EVICT=lfu"
trap 'docker rm -f "$name" >/dev/null 2>&1' EXIT
phase() {  # tag pin_gib bench_args serve_flags...; env KS="KSTAGE=... VAR=..." loads probes/kstage
  local tag=$1 pin=$2 w=$3; shift 3
  local envx=()
  if [ -n "${KS:-}" ]; then
    envx+=(-v "$here/probes/kstage":/k:ro -v "$here/profiles":/prof:ro -e PYTHONPATH=/k)
    for v in $KS; do envx+=(-e "$v"); done
  fi
  echo "== phase $tag $(date -u +%H:%M:%SZ) work: agent${KS:+ ks: $KS}"
  # Pinned host memory cannot be swapped: on 30 GB beside the monitoring stack, skip an arm
  # whose pins (host KV, K6's expert homes) would leave under 4 GiB.
  local avail=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
  if [ "$avail" -lt $((pin + 4)) ]; then echo "  skipped: pins ${pin} GiB, ${avail} GiB available"; return 4; fi
  if sudo -n true 2>/dev/null; then   # K6 pins in 2 MiB pieces: drop the clean cache and compact
    sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'
  fi
  local ram0=$(free -m | awk '/^Mem:/{print $3}')
  docker run -d --name "$name" --gpus all --ipc=host -p 127.0.0.1:8323:8000 \
    -v /srv/model-cache:/hf:ro -v "$adapter":/adapter:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 "${envx[@]}" \
    --pull never vllm/vllm-openai:v0.29.0 \
    --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --revision bee7596271d1495f6992ae224aefde4410e816b8 --served-model-name lightning-nvfp4 \
    --kv-cache-dtype fp8 --mamba-cache-mode align --moe-backend marlin --linear-backend marlin \
    --max-model-len 131072 --max-num-seqs 16 --gpu-memory-utilization 0.92 \
    "${LORA[@]}" "${TM[@]}" --enable-prompt-tokens-details "$@" >/dev/null || return 1
  local t0=$(date +%s)
  until curl -sf $B/health >/dev/null; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
      echo "container exited before ready ($(( $(date +%s)-t0 ))s)"; docker logs "$name" > "$J/serve-$tag.log" 2>&1
      docker rm -f "$name" >/dev/null 2>&1; return 2; fi
    [ $(( $(date +%s)-t0 )) -gt 900 ] && { echo "not ready in 900s"; docker logs "$name" > "$J/serve-$tag.log" 2>&1; docker rm -f "$name" >/dev/null 2>&1; return 3; }
    sleep 5
  done
  echo "ready in $(( $(date +%s)-t0 ))s; $(docker logs "$name" 2>&1 | grep -oE "Model loading took [0-9.]+ GiB|Available KV cache memory: [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens" | tr '\n' ';')"
  echo "  backends: $(docker logs "$name" 2>&1 | grep -oE "Using [A-Za-z0-9_]+ (attention )?backend[^,;]*|Selected [A-Za-z0-9_]+ for [A-Za-z ]+" | sort -u | tr '\n' ';' | cut -c1-300)"
  [ -z "${KS:-}" ] || docker logs "$name" 2>&1 | grep -E "kstage: (cache|installed|registered)" | sed 's/^.*kstage:/  kstage:/' | sort -u
  echo "  host RAM used: +$(( $(free -m | awk '/^Mem:/{print $3}') - ram0 )) MiB, $(free -m | awk '/^Mem:/{print $7}') MiB available"
  local t=$(date +%s)
  # The client gets 40 minutes; a stall ends the bench, never the container (its log is kept)
  # shellcheck disable=SC2086  # w is a word list
  timeout 2400 python3 "$here/probes/offload_bench.py" --base $B --model g6q --out "$J/bench-$tag.json" agent $w \
    > "$J/bench-$tag.log" 2>&1
  echo "  bench: exit $?, $(( $(date +%s)-t ))s"; grep -E "^  agent" "$J/bench-$tag.log"
  docker logs "$name" > "$J/serve-$tag.log" 2>&1
  echo "  max running/waiting: $(grep -oE "Running: [0-9]+ reqs, Waiting: [0-9]+ reqs" "$J/serve-$tag.log" | sort -t' ' -k2,2n -k5,5n | tail -1)"
  [ -z "${KS:-}" ] || grep -E "kstage: cache:" "$J/serve-$tag.log" | tail -2 | sed 's/^.*kstage:/  kstage:/'
  docker rm -f "$name" >/dev/null 2>&1
}
run() {  # d41-{base|cgN}[-kvoffN][-bNk][-kvbf16]-a8l|a16l
  local o=() w=$AGENT8L pin=0 k=$KC6
  case $1 in *-base*) k="" ;; *) pin=17 ;; esac   # K6 with every slot homes all experts: ~16.5 GiB
  case $1 in *-cg*) local g=${1#*-cg}; k="${k/KSTAGE_COLD_GB=4/KSTAGE_COLD_GB=${g%%-*}}" ;; esac
  case $1 in *-kvoff8*) o=(--kv-offloading-size 8 --kv-offloading-backend native); pin=$((pin + 8)) ;;
             *-kvoff16*) o=(--kv-offloading-size 16 --kv-offloading-backend native); pin=$((pin + 16)) ;; esac
  case $1 in *-b[0-9]*k-*) local b=${1##*-b}; o+=(--max-num-batched-tokens $((${b%%k*} * 1024))) ;; esac
  case $1 in *-kvbf16*) o+=(--kv-cache-dtype bfloat16) ;; esac
  case $1 in *-a16l) w=$AGENT16L ;; esac
  KS="$k" phase "$1" "$pin" "$w" "${o[@]}"
}
for p in "$@"; do
  case $p in
    d41) for q in d41-base-kvoff8-b4k-a8l d41-base-kvoff8-b8k-a8l d41-base-b4k-a8l d41-k6-b8k-a8l; do run $q; done ;;
    *) run "$p" ;;
  esac
done

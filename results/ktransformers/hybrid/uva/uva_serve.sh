#!/bin/bash
# Desktop (3090): Lightning NVFP4 with $GB GiB of experts on vLLM's UVA offload, stock pinned pages
# vs KSTAGE_UVA_THP=1 (2 MiB pages), decode at 1/4/16 streams, ABBA. No adapter. Runs from
# ~/kt-relu2-results/uvathp (k/ = kstage, offload_bench.py beside it). Port 127.0.0.1:8324.
D=$HOME/kt-relu2-results/uvathp; GB=${1:-4}
name=kuva-tm; B=http://127.0.0.1:8324
trap 'docker rm -f "$name" >/dev/null 2>&1' EXIT
phase() {  # tag [extra env VAR=value...]
  local tag=$1; shift
  local envx=(-v "$D/k":/k:ro -e PYTHONPATH=/k)
  for v in "$@"; do envx+=(-e "$v"); done
  echo "== phase $tag $(date -u +%H:%M:%SZ) offload ${GB} GiB env: $*"
  sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'
  local ram0=$(free -m | awk '/^Mem:/{print $3}')
  docker run -d --name "$name" --gpus all --ipc=host -p 127.0.0.1:8324:8000 \
    -v /srv/model-cache:/hf:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 "${envx[@]}" \
    --pull never vllm/vllm-openai:v0.29.0 \
    --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --revision bee7596271d1495f6992ae224aefde4410e816b8 --served-model-name lightning-nvfp4 \
    --kv-cache-dtype fp8 --mamba-cache-mode align --moe-backend marlin --linear-backend marlin \
    --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.85 \
    --cpu-offload-params experts --cpu-offload-gb "$GB" --enable-prompt-tokens-details >/dev/null || return 1
  local t0=$(date +%s)
  until curl -sf $B/health >/dev/null; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
      echo "container exited before ready ($(( $(date +%s)-t0 ))s)"; docker logs "$name" > "$D/serve-$tag.log" 2>&1
      docker rm -f "$name" >/dev/null 2>&1; return 2; fi
    [ $(( $(date +%s)-t0 )) -gt 900 ] && { echo "not ready in 900s"; docker logs "$name" > "$D/serve-$tag.log" 2>&1; docker rm -f "$name" >/dev/null 2>&1; return 3; }
    sleep 5
  done
  echo "ready in $(( $(date +%s)-t0 ))s; $(docker logs "$name" 2>&1 | grep -oE "Total CPU offloaded parameters: [0-9.]+ GiB|Available KV cache memory: [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens" | tr '\n' ';')"
  docker logs "$name" 2>&1 | grep -E "kstage:" | sed 's/^.*kstage:/  kstage:/' | sort -u
  echo "  host RAM used: +$(( $(free -m | awk '/^Mem:/{print $3}') - ram0 )) MiB, $(free -m | awk '/^Mem:/{print $7}') MiB available; AnonHugePages $(awk '/AnonHugePages/{print $2/1048576 " GiB"}' /proc/meminfo)"
  local t=$(date +%s)
  python3 "$D/offload_bench.py" --base $B --model lightning-nvfp4 --out "$D/bench-$tag.json" decode --conc 1,4,16 \
    > "$D/bench-$tag.log" 2>&1
  echo "  bench: exit $?, $(( $(date +%s)-t ))s"; grep -E "^  decode" "$D/bench-$tag.log"
  docker logs "$name" > "$D/serve-$tag.log" 2>&1
  docker rm -f "$name" >/dev/null 2>&1
}
THP=(KSTAGE_UVA_THP=1 VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1)
phase uva$GB-base-r1
phase uva$GB-thp-r1 "${THP[@]}"
phase uva$GB-thp-r2 "${THP[@]}"
phase uva$GB-base-r2
echo "== done $(date -u +%H:%M:%SZ)"

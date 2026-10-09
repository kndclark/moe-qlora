#!/bin/bash
# KSTAGE_THP end to end: p20's live copy + LFU (419 tok/s at 16 streams) with and without the
# 2 MiB-page mirror of the host rows, decode at 1/4/16 streams, ABBA. Mirrors kv_levers.sh
# phase() for the decode path; its own container name and port, never touches kvlever-tm.
H=/home/david/handoff-0948db4b-files/hybrid/thp
here=/home/david/moe-qlora/.claude/worktrees/next-round
adapter=/home/david/moe-qlora/results/g6q-train-adapter
name=kthp-tm; B=http://127.0.0.1:8323
KS="KSTAGE=cache KSTAGE_COLD_GB=4 KSTAGE_PROFILE=/prof/p7-all.json KSTAGE_STATS=1 KSTAGE_SLOTS=all"
KS="$KS KSTAGE_DMA_M=64 KSTAGE_DMA_BUF=1 KSTAGE_COPY_LIVE=82 KSTAGE_EVICT=lfu"
restore() {
  docker rm -f "$name" >/dev/null 2>&1
  echo performance | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null
  echo "profile: $(cat /sys/firmware/acpi/platform_profile)"
}
trap restore EXIT
echo max-power | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null
phase() {  # tag [extra KSTAGE vars]
  local tag=$1; shift
  local envx=(-v "$here/probes/kstage":/k:ro -v "$here/results/kv-levers/profiles":/prof:ro -e PYTHONPATH=/k)
  for v in $KS "$@"; do envx+=(-e "$v"); done
  echo "== phase $tag $(date -u +%H:%M:%SZ) ks: $KS $*"
  sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'
  awk '/Normal/{for(i=14;i<=NF;i++)s+=$i*2^(i-5)/256} END{printf "  free in 2 MiB+ blocks: %.1f GiB\n", s/1024}' /proc/buddyinfo
  local ram0=$(free -m | awk '/^Mem:/{print $3}')
  docker run -d --name "$name" --gpus all --ipc=host -p 127.0.0.1:8323:8000 \
    -v /srv/model-cache:/hf:ro -v "$adapter":/adapter:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 "${envx[@]}" \
    --pull never vllm/vllm-openai:v0.29.0 \
    --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --revision bee7596271d1495f6992ae224aefde4410e816b8 --served-model-name lightning-nvfp4 \
    --kv-cache-dtype fp8 --mamba-cache-mode align --moe-backend marlin \
    --max-model-len 131072 --max-num-seqs 16 --gpu-memory-utilization 0.85 \
    --enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules "g6q=/adapter" \
    --lora-target-modules q_proj k_proj v_proj o_proj in_proj up_proj down_proj \
    --enable-prompt-tokens-details --gpu-memory-utilization 0.92 >/dev/null || return 1
  local t0=$(date +%s)
  until curl -sf $B/health >/dev/null; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
      echo "container exited before ready ($(( $(date +%s)-t0 ))s)"; docker logs "$name" > "$H/serve-$tag.log" 2>&1
      docker rm -f "$name" >/dev/null 2>&1; return 2; fi
    [ $(( $(date +%s)-t0 )) -gt 900 ] && { echo "not ready in 900s"; docker logs "$name" > "$H/serve-$tag.log" 2>&1; docker rm -f "$name" >/dev/null 2>&1; return 3; }
    sleep 5
  done
  echo "ready in $(( $(date +%s)-t0 ))s; $(docker logs "$name" 2>&1 | grep -oE "Available KV cache memory: [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens" | tr '\n' ';')"
  docker logs "$name" 2>&1 | grep -E "kstage: (cache|installed|registered)" | sed 's/^.*kstage:/  kstage:/' | sort -u
  echo "  host RAM used: +$(( $(free -m | awk '/^Mem:/{print $3}') - ram0 )) MiB, $(free -m | awk '/^Mem:/{print $7}') MiB available"
  local t=$(date +%s)
  python3 "$here/probes/offload_bench.py" --base $B --model g6q --out "$H/bench-$tag.json" decode --conc 1,4,16 \
    > "$H/bench-$tag.log" 2>&1
  echo "  bench: exit $?, $(( $(date +%s)-t ))s"; grep -E "^  decode" "$H/bench-$tag.log"
  docker logs "$name" > "$H/serve-$tag.log" 2>&1
  echo "  max running/waiting: $(grep -oE "Running: [0-9]+ reqs, Waiting: [0-9]+ reqs" "$H/serve-$tag.log" | sort -t' ' -k2,2n -k5,5n | tail -1)"
  docker rm -f "$name" >/dev/null 2>&1
}
phase thp-base-r1
phase thp-on-r1 KSTAGE_THP=1
phase thp-on-r2 KSTAGE_THP=1
phase thp-base-r2
echo "== done $(date -u +%H:%M:%SZ)"

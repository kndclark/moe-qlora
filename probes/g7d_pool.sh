#!/bin/bash
# G7d (plan.md "G7d"): the G6u adapter served on the pool (both cards, PP 2) with G7c2's
# settings plus G7b's LoRA flags. The adapter's two files sit at
# /srv/model-cache/adapters/lightning-g6u on both nodes (desktop cache, laptop mirror).
# Runs G7b's seven thinking-off sets, then G7b-think's five thinking-on sets, through
# probes/g7a_eval.py against the pool. Always ends with `lab pool down` and then `lab up`
# on the desktop, which pool down does not do.
# Outputs: results/g7d/research-eval-<tag>-lightning-g6u-{nothink,think-4k}-pool.{json,log},
# results/g7d/{pool-up,vllm-pool,pool-down,desktop-lab-up}.log.
# usage: g7d_pool.sh
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
out=$here/results/g7d
mkdir -p "$out"
LAB=/home/david/gpu-lab/bin/lab
M=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
B=http://lab-desktop:8200
LABEL=g6u
python3 "$here/probes/g7a_eval.py" --selfcheck | tail -1 | grep -qx "selfcheck PASS" || { echo "g7a_eval.py --selfcheck failed"; exit 4; }
finish() {
  ssh llm "sudo docker exec ray-head cat /tmp/vllm-pool.log" > "$out/vllm-pool.log" 2>&1
  "$LAB" pool down > "$out/pool-down.log" 2>&1; echo "pool down exit $?"
  ssh llm "$LAB up" > "$out/desktop-lab-up.log" 2>&1
  echo "desktop lab up exit $?; llama-swap $(ssh llm systemctl is-active llama-swap)"
}
trap finish EXIT
POOL_MODEL=$M POOL_MAXLEN=16384 POOL_WAIT_SECS=900 POOL_GPU_UTIL=0.85 \
POOL_EXTRA_FLAGS="--revision bee7596271d1495f6992ae224aefde4410e816b8 --kv-cache-dtype fp8 --mamba-cache-mode align --moe-backend marlin --linear-backend marlin --enforce-eager --max-num-seqs 16 --enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules $LABEL=/hf/adapters/lightning-g6u" \
  "$LAB" pool up > "$out/pool-up.log" 2>&1
rc=$?
echo "pool up exit $rc $(date +%T)"
if [ $rc -ne 0 ]; then
  ssh llm "sudo docker exec ray-head cat /tmp/vllm-pool.log" 2>/dev/null | grep -E "Error|error" | tail -4 | cut -c1-240
  exit 1
fi
ssh llm "sudo docker exec ray-head grep -E 'KV cache size|Maximum concurrency|Available KV' /tmp/vllm-pool.log" | sed 's/^.*INFO/INFO/' | cut -c1-200
echo "models: $(curl -s $B/v1/models | python3 -c 'import json,sys;print(*[m["id"] for m in json.load(sys.stdin)["data"]])')"
for mode in off on; do
  if [ $mode = off ]; then
    common=(--max-calls 3 --temperature 0 --window 4000 --seed 20260923 --concurrency 16 --thinking off --max-tokens 512)
    sets=(v1:v1 v2:v2 rocky:rocky promqlcat:promql general:general alert:alert trap3:trap3)
    suffix=nothink
  else
    common=(--max-calls 3 --temperature 0 --window 4000 --seed 20260923 --concurrency 16 --thinking on --max-tokens 4096)
    sets=(v2:v2 rocky:rocky promqlcat:promql alert:alert trap3:trap3)
    suffix=think-4k
  fi
  for s in "${sets[@]}"; do
    tag=${s%%:*} set=${s##*:} extra=()
    [ "$tag" = promqlcat ] && extra=(--promql-catalog)
    label=$tag-lightning-$LABEL-$suffix-pool
    t=$(date +%s)
    python3 "$here/probes/g7a_eval.py" --base $B --model "$LABEL" --label "$label" --set "$set" \
      "${common[@]}" "${extra[@]}" --out "$out/research-eval-$label.json" > "$out/research-eval-$label.log" 2>&1
    echo "  $label: exit $?, $(( $(date +%s)-t ))s"
  done
done
echo "GPUs after: laptop $(nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader); desktop $(ssh llm 'nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader')"

#!/bin/bash
# N1 (plan.md "N1"): repeat the thinking-on eval runs of the contested sets, on a server
# started exactly as probes/g6_eval.sh starts it (same image, flags, one LoRA), to measure
# run-to-run spread at the eval's own settings (temperature 0, concurrency 16).
# Outputs go to results/noise/, never over the gate's files.
# usage: [THINK=off] LABEL=g6q noise_eval.sh ADAPTER_DIR "r2 r3" [base]
#   "base" also repeats base Lightning on this server, as g6_eval.sh ran base.
#   THINK=off (N2): g6_eval.sh's thinking-off settings on all seven sets, labels ...-nothink-rN.
set -u
adapter=$(realpath "$1"); reps=$2; with_base=${3:-}
here=$(cd "$(dirname "$0")/.." && pwd)
out=$here/results/noise
mkdir -p "$out"
LABEL=${LABEL:?LABEL names the adapter}
name=$LABEL-noise
slog=$out/$LABEL-noise-serve.log
[ "${THINK:-on}" = off ] && slog=$out/$LABEL-noise-nothink-serve.log  # N2 must not overwrite N1's
B=http://127.0.0.1:8303
python3 "$here/probes/g7a_eval.py" --selfcheck | tail -1 | grep -qx "selfcheck PASS" || { echo "g7a_eval.py --selfcheck failed"; exit 4; }
restore() {
  docker logs "$name" > "$slog" 2>&1 || true
  docker rm -f "$name" >/dev/null 2>&1
  echo performance | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null
  echo "profile: $(cat /sys/firmware/acpi/platform_profile)"
}
trap restore EXIT
echo max-power | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null
docker run -d --name "$name" --gpus all --ipc=host -p 127.0.0.1:8303:8000 \
  -v /srv/model-cache:/hf:ro -v "$adapter":/adapter:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai:v0.29.0 \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --revision bee7596271d1495f6992ae224aefde4410e816b8 --served-model-name lightning-nvfp4 \
  --kv-cache-dtype fp8 --mamba-cache-mode align --moe-backend marlin \
  --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.85 --enforce-eager \
  --enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules "$LABEL=/adapter" >/dev/null || exit 1
t0=$(date +%s)
until curl -sf $B/health >/dev/null; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
    echo "container exited before ready ($(( $(date +%s)-t0 ))s)"; exit 2; fi
  [ $(( $(date +%s)-t0 )) -gt 600 ] && { echo "not ready in 600s"; exit 3; }
  sleep 5
done
echo "ready in $(( $(date +%s)-t0 ))s"
common=(--max-calls 3 --temperature 0 --window 4000 --seed 20260923 --concurrency 16 --thinking on --max-tokens 4096)
mode=think-4k
sets=(v2:v2 rocky:rocky promqlcat:promql alert:alert trap3:trap3)
if [ "${THINK:-on}" = off ]; then
  common=(--max-calls 3 --temperature 0 --window 4000 --seed 20260923 --concurrency 16 --thinking off --max-tokens 512)
  mode=nothink
  sets=(v1:v1 v2:v2 rocky:rocky promqlcat:promql general:general alert:alert trap3:trap3)
fi
run() {  # label model set [extra...]
  local label=$1 model=$2 set=$3; shift 3
  if [ -f "$out/research-eval-$label.json" ]; then echo "  $label: exists, skipped"; return; fi
  if [ "$set" = promql ] && ! curl -sf -m3 http://lab-desktop:9090/-/ready >/dev/null; then
    echo "  $label: SKIPPED, desktop Prometheus unreachable"; return; fi
  local t=$(date +%s)
  python3 "$here/probes/g7a_eval.py" --base $B --model "$model" --label "$label" --set "$set" \
    "${common[@]}" "$@" --out "$out/research-eval-$label.json" > "$out/research-eval-$label.log" 2>&1
  echo "  $label: exit $?, $(( $(date +%s)-t ))s"
}
for rep in $reps; do
  echo "== $rep"
  for s in "${sets[@]}"; do
    tag=${s%%:*} set=${s##*:} extra=()
    [ "$tag" = promqlcat ] && extra=(--promql-catalog)
    run "$tag-lightning-$LABEL-$mode-$rep" "$LABEL" "$set" "${extra[@]}"
    [ "$with_base" = base ] && run "$tag-lightning-$mode-$rep" lightning-nvfp4 "$set" "${extra[@]}"
  done
done

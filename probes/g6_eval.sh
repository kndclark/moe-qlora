#!/bin/bash
# G6 eval (plan.md G6): base Lightning NVFP4 and the G6 adapter, served together by one
# vLLM server (G7a's flags plus LoRA, as g7_lora_smoke.sh), on the lab's seven eval sets.
#   primary:   thinking on, 4096 tokens: adapter and base, both on this server
#   secondary: thinking off, G7a's settings: adapter here; base is G7a's runs
#   checks:    v1 base again on this server, both modes, against G7a's non-LoRA server
# Harness: probes/g7a_eval.py, G7a's wrapper of gpu-lab bench/research_eval.py that reads
# Lightning's XML tool calls and its open <think>; the bare harness parses neither
# (first attempt, results/g6-eval-attempt1-unpatched-harness.log). Its --selfcheck gates the run.
# Existing outputs are skipped, so a rerun resumes. The promql set asks the desktop's
# Prometheus over the direct link; it is skipped, and says so, if the link is down.
# usage: [LABEL=g6r] g6_eval.sh ADAPTER_DIR   LABEL names the adapter and its outputs
# (default g6, G6 as run); base outputs keep their names, so a new LABEL reuses them.
set -u
adapter=$(realpath "$1")
here=$(cd "$(dirname "$0")/.." && pwd)
out=$here/results
LABEL=${LABEL:-g6}
name=$LABEL-eval
B=http://127.0.0.1:8303
python3 "$here/probes/g7a_eval.py" --selfcheck | tail -1 | grep -qx "selfcheck PASS" || { echo "g7a_eval.py --selfcheck failed"; exit 4; }
restore() {
  docker logs "$name" > "$out/$LABEL-eval-serve.log" 2>&1 || true
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
echo "ready in $(( $(date +%s)-t0 ))s; models: $(curl -s $B/v1/models | python3 -c 'import json,sys;print(*[m["id"] for m in json.load(sys.stdin)["data"]])')"

common=(--max-calls 3 --temperature 0 --window 4000 --seed 20260923 --concurrency 16)
run() {  # label model thinking max_tokens set [extra...]
  local label=$1 model=$2 think=$3 mt=$4 set=$5; shift 5
  if [ -f "$out/research-eval-$label.json" ]; then echo "  $label: exists, skipped"; return; fi
  if [ "$set" = promql ] && ! curl -sf -m3 http://lab-desktop:9090/-/ready >/dev/null; then
    echo "  $label: SKIPPED, desktop Prometheus unreachable (link down?)"; return; fi
  local t=$(date +%s)
  python3 "$here/probes/g7a_eval.py" --base $B --model "$model" --label "$label" \
    --thinking "$think" --max-tokens "$mt" --set "$set" "${common[@]}" "$@" \
    --out "$out/research-eval-$label.json" > "$out/research-eval-$label.log" 2>&1
  echo "  $label: exit $?, $(( $(date +%s)-t ))s"
}
sets=(v1:v1 v2:v2 rocky:rocky promqlcat:promql general:general alert:alert trap3:trap3)
for mode in adapter-think base-think adapter-nothink; do
  echo "== $mode"
  for s in "${sets[@]}"; do
    tag=${s%%:*} set=${s##*:} extra=()
    [ "$tag" = promqlcat ] && extra=(--promql-catalog)
    case $mode in
      adapter-think)   run "$tag-lightning-$LABEL-think-4k" "$LABEL" on 4096 "$set" "${extra[@]}" ;;
      base-think)      l="$tag-lightning-think-4k"; [ "$tag" = v1 ] && l=v1-lightning-think-4k-g6srv
                       run "$l" lightning-nvfp4 on 4096 "$set" "${extra[@]}" ;;
      adapter-nothink) run "$tag-lightning-$LABEL-nothink" "$LABEL" off 512 "$set" "${extra[@]}" ;;
    esac
  done
done
echo "== checks"
run v1-lightning-nothink-g6srv lightning-nvfp4 off 512 v1

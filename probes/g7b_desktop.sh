#!/bin/bash
# G7b (plan.md "G7b"): Lightning NVFP4 + an adapter served on the desktop's 3090 (sm_86),
# with probes/g6_eval.sh's server flags, bound to the direct link only; the seven
# thinking-off sets run from the laptop through probes/g7a_eval.py, as the gate runs them.
# Outputs: results/g7b/research-eval-<tag>-lightning-<LABEL>-nothink-desk.{json,log},
# results/g7b/<LABEL>-desk-serve.log.
# usage: LABEL=g6u [DROP="--kv-cache-dtype fp8"] [EXTRA="--linear-backend marlin"] g7b_desktop.sh ADAPTER_DIR
#   DROP removes one flag the server refuses on sm_86 (plan.md: at most one, and said so).
#   EXTRA adds server flags (plan.md "G7b" amendment: --linear-backend marlin).
#   THINK=on (G7b-think): the gate's thinking-on settings on N1's five sets, labels ...-think-4k-desk.
set -u
adapter=$(realpath "$1")
LABEL=${LABEL:?LABEL names the adapter}
here=$(cd "$(dirname "$0")/.." && pwd)
out=$here/results/g7b
mkdir -p "$out"
name=g7b-$LABEL
B=http://lab-desktop:8303
remote=/home/david/g7b/$LABEL
python3 "$here/probes/g7a_eval.py" --selfcheck | tail -1 | grep -qx "selfcheck PASS" || { echo "g7a_eval.py --selfcheck failed"; exit 4; }
ssh llm "mkdir -p $remote" && scp -q -r "$adapter"/adapter_config.json "$adapter"/adapter_model.safetensors llm:$remote/ || exit 1
flags="--kv-cache-dtype fp8 --mamba-cache-mode align --moe-backend marlin"
[ -n "${DROP:-}" ] && flags=${flags/$DROP/}
flags="$flags ${EXTRA:-}"
slog=$out/$LABEL-desk-serve.log
[ "${THINK:-off}" = on ] && slog=$out/$LABEL-desk-think-serve.log  # never over the thinking-off log
restore() {
  ssh llm "sudo docker logs $name" > "$slog" 2>&1 || true
  ssh llm "sudo docker rm -f $name" >/dev/null 2>&1
}
trap restore EXIT
ssh llm "sudo docker run -d --name $name --gpus all --ipc=host -p lab-desktop:8303:8000 \
  -v /srv/model-cache:/hf:ro -v $remote:/adapter:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai:v0.29.0 \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --revision bee7596271d1495f6992ae224aefde4410e816b8 --served-model-name lightning-nvfp4 \
  $flags --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.85 --enforce-eager \
  --enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules $LABEL=/adapter" >/dev/null || exit 1
echo "flags: $flags"
t0=$(date +%s)
until curl -sf $B/health >/dev/null; do
  if [ "$(ssh llm "sudo docker inspect -f '{{.State.Running}}' $name" 2>/dev/null)" != true ]; then
    echo "container exited before ready ($(( $(date +%s)-t0 ))s)"; exit 2; fi
  [ $(( $(date +%s)-t0 )) -gt 900 ] && { echo "not ready in 900s"; exit 3; }
  sleep 10
done
echo "ready in $(( $(date +%s)-t0 ))s; models: $(curl -s $B/v1/models | python3 -c 'import json,sys;print(*[m["id"] for m in json.load(sys.stdin)["data"]])')"
echo "desktop GPU: $(ssh llm 'nvidia-smi --query-gpu=memory.used,temperature.gpu,power.draw --format=csv,noheader')"
common=(--max-calls 3 --temperature 0 --window 4000 --seed 20260923 --concurrency 16 --thinking off --max-tokens 512)
sets=(v1:v1 v2:v2 rocky:rocky promqlcat:promql general:general alert:alert trap3:trap3)
mode=nothink
if [ "${THINK:-off}" = on ]; then
  common=(--max-calls 3 --temperature 0 --window 4000 --seed 20260923 --concurrency 16 --thinking on --max-tokens 4096)
  sets=(v2:v2 rocky:rocky promqlcat:promql alert:alert trap3:trap3)
  mode=think-4k
fi
for s in "${sets[@]}"; do
  tag=${s%%:*} set=${s##*:} extra=()
  [ "$tag" = promqlcat ] && extra=(--promql-catalog)
  label=$tag-lightning-$LABEL-$mode-desk
  t=$(date +%s)
  python3 "$here/probes/g7a_eval.py" --base $B --model "$LABEL" --label "$label" --set "$set" \
    "${common[@]}" "${extra[@]}" --out "$out/research-eval-$label.json" > "$out/research-eval-$label.log" 2>&1
  echo "  $label: exit $?, $(( $(date +%s)-t ))s"
done
echo "desktop GPU after: $(ssh llm 'nvidia-smi --query-gpu=memory.used,temperature.gpu,power.draw --format=csv,noheader')"

#!/bin/bash
# N2 (plan.md "N2"): repeat the Qwen3-8B v3 adapter's thinking-off yardstick runs
# (~/gpu-lab/bench/results/research-eval-*-L-adv3-nothink.json, g6_compare.py's "vs Qwen"),
# on the server those runs used: session 09256598's docker command, with the adapters the
# r1 provenance lists (research-v2, research-v3, research-v3-ep1). Harness: gpu-lab's
# bench/research_eval.py as it is now; r1 ran an earlier, uncommitted version, and
# rescoring r1 with the current scorer changes no headline row (plan.md "N2").
# Outputs: results/noise/research-eval-<tag>-L-adv3-nothink-<rep>.json (v1 has no tag).
# usage: [THINK=on] n2_qwen.sh "r2 r3"
#   THINK=on (gap ledger L8): thinking on, 4096 tokens, N1's five sets, labels
#   ...-L-adv3-think-4k-rN, and --max-model-len 16384 (G6u's server) so the answers fit.
set -u
reps=$1
here=$(cd "$(dirname "$0")/.." && pwd)
out=$here/results/noise
mkdir -p "$out"
H=$HOME/gpu-lab/bench/research_eval.py
name=qwen3-8b-n2
B=http://127.0.0.1:8300
maxlen=8192 mode=nothink
[ "${THINK:-off}" = on ] && maxlen=16384 mode=think-4k
restore() {
  docker logs "$name" > "$out/qwenv3-noise-serve.log" 2>&1 || true
  docker rm -f "$name" >/dev/null 2>&1
  echo performance | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null
  echo "profile: $(cat /sys/firmware/acpi/platform_profile)"
}
trap restore EXIT
echo max-power | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null
docker run -d --name "$name" --init --gpus all --ipc=host -v /srv/model-cache:/hf \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -p 127.0.0.1:8300:8000 vllm/vllm-openai:v0.29.0 \
  --model Qwen/Qwen3-8B --enable-lora --max-lora-rank 16 --max-loras 4 \
  --lora-modules research-v2=/hf/adapters/qwen3-8b-research-v2 \
  research-v3=/hf/adapters/qwen3-8b-research-v3 \
  research-v3-ep1=/hf/adapters/qwen3-8b-research-v3/checkpoints/checkpoint-119 \
  --max-model-len $maxlen --gpu-memory-utilization 0.85 >/dev/null || exit 1
t0=$(date +%s)
until curl -sf $B/health >/dev/null; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
    echo "container exited before ready ($(( $(date +%s)-t0 ))s)"; exit 2; fi
  [ $(( $(date +%s)-t0 )) -gt 600 ] && { echo "not ready in 600s"; exit 3; }
  sleep 5
done
echo "ready in $(( $(date +%s)-t0 ))s; models: $(curl -s $B/v1/models | python3 -c 'import json,sys;print(*[m["id"] for m in json.load(sys.stdin)["data"]])')"
common=(--max-calls 3 --temperature 0 --window 4000 --seed 20260923 --concurrency 16 --thinking off --max-tokens 512)
sets=(v1:v1 v2:v2 rocky:rocky promqlcat:promql general:general alert:alert trap3:trap3)
if [ $mode = think-4k ]; then
  common=(--max-calls 3 --temperature 0 --window 4000 --seed 20260923 --concurrency 16 --thinking on --max-tokens 4096)
  sets=(v2:v2 rocky:rocky promqlcat:promql alert:alert trap3:trap3)
fi
for rep in $reps; do
  echo "== $rep"
  for s in "${sets[@]}"; do
    tag=${s%%:*} set=${s##*:} extra=()
    [ "$tag" = promqlcat ] && extra=(--promql-catalog)
    label=$tag-L-adv3-$mode-$rep
    [ "$tag" = v1 ] && label=L-adv3-$mode-$rep
    if [ -f "$out/research-eval-$label.json" ]; then echo "  $label: exists, skipped"; continue; fi
    if [ "$set" = promql ] && ! curl -sf -m3 http://lab-desktop:9090/-/ready >/dev/null; then
      echo "  $label: SKIPPED, desktop Prometheus unreachable"; continue; fi
    t=$(date +%s)
    python3 "$H" --base $B --model research-v3 --label "$label" --set "$set" "${common[@]}" \
      "${extra[@]}" --out "$out/research-eval-$label.json" > "$out/research-eval-$label.log" 2>&1
    echo "  $label: exit $?, $(( $(date +%s)-t ))s"
  done
done

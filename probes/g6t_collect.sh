#!/bin/bash
# G6t step 1: serve base Lightning NVFP4 exactly as probes/g6_eval.sh does, minus the LoRA
# flags, and run probes/g6t_collect.py against it. The promql records need the desktop's
# Prometheus over the direct link; the run refuses to start without it.
# usage: [ADAPTER=dir LORA=name] g6t_collect.sh OUT.jsonl [g6t_collect.py args...]
#   with ADAPTER, the adapter is served as g6_eval.sh serves it (pass --model NAME too).
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
addr=$(ssh -G llm | awk '/^hostname /{print $2}')   # the desktop's end of the direct link
out=$(realpath -m "$1"); shift
name=g6t-collect
B=http://127.0.0.1:8303
curl -sf -m3 http://$addr:9090/-/ready >/dev/null || { echo "desktop Prometheus unreachable"; exit 5; }
python3 "$here/probes/g7a_eval.py" --selfcheck | tail -1 | grep -qx "selfcheck PASS" || { echo "g7a_eval.py --selfcheck failed"; exit 4; }
restore() {
  docker logs "$name" > "${out%.jsonl}-serve.log" 2>&1 || true
  docker rm -f "$name" >/dev/null 2>&1
  echo performance | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null
  echo "profile: $(cat /sys/firmware/acpi/platform_profile)"
}
trap restore EXIT
echo max-power | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null
lora=() mount=()
if [ -n "${ADAPTER:-}" ]; then
  mount=(-v "$(realpath "$ADAPTER")":/adapter:ro)
  lora=(--enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules "${LORA:?LORA name needed}=/adapter")
fi
docker run -d --name "$name" --gpus all --ipc=host -p 127.0.0.1:8303:8000 \
  -v /srv/model-cache:/hf:ro "${mount[@]}" -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai:v0.29.0 \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --revision bee7596271d1495f6992ae224aefde4410e816b8 --served-model-name lightning-nvfp4 \
  --kv-cache-dtype fp8 --mamba-cache-mode align --moe-backend marlin \
  --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.85 --enforce-eager "${lora[@]}" >/dev/null || exit 1
t0=$(date +%s)
until curl -sf $B/health >/dev/null; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
    echo "container exited before ready ($(( $(date +%s)-t0 ))s)"; exit 2; fi
  [ $(( $(date +%s)-t0 )) -gt 600 ] && { echo "not ready in 600s"; exit 3; }
  sleep 5
done
echo "ready in $(( $(date +%s)-t0 ))s"
t=$(date +%s)
python3 "$here/probes/g6t_collect.py" --base $B --out "$out" "$@"
echo "collect exit $?, $(( $(date +%s)-t ))s"

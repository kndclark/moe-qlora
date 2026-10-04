#!/bin/bash
# O70 (docs/next-model-plan.md): the 4-bit Llama-3.1-70B served by vLLM on the laptop's one
# card, with part of its decoder layers in pinned RAM. One arm per run: ARM names it and
# OFFLOAD holds its vLLM offload flags. Everything else is the pool's L70 serving (maxlen
# 8192, util 0.92, a 512-token prefill batch, no FlashInfer autotune), eager unless EAGER=0.
# The pool must be down: the laptop's card is its second stage.
# Records under results/o70/<ARM>/: serve.log, kv.txt, mem.txt, bench-c{1,4,16}.{json,log};
#   with EVAL=1 the seven sets too (l70_eval.sh, TAG=o70-<ARM>, Meta's tool format).
# usage: ARM=uva OFFLOAD="--cpu-offload-gb 21.5" [EAGER=1] [EVAL=1] o70_serve.sh
set -u
ARM=${ARM:?names the arm} OFFLOAD=${OFFLOAD:?vLLM offload flags}
MODEL=hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4 REV=1b0ae7f9d6da8b79f36fdc24912f950ecb2b6e91
B=http://127.0.0.1:8305 name=o70-$ARM
here=$(cd "$(dirname "$0")/.." && pwd)
out=$here/results/o70/$ARM
mkdir -p "$out"
# The terminal (ptyxis) is the one GPU client allowed: vLLM charges the others' memory to KV.
apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -v ptyxis)
[ -n "$apps" ] && { echo "laptop GPU has other clients (pool up? apps open?):"; echo "$apps"; exit 2; }
stop() {
  docker logs $name > "$out/serve.log" 2>&1 || true
  docker rm -f $name >/dev/null 2>&1
  echo performance | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null
  echo "profile: $(cat /sys/firmware/acpi/platform_profile)"
}
trap stop EXIT
echo max-power | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null
eager=(--enforce-eager); [ "${EAGER:-1}" = 0 ] && eager=()
docker run -d --name $name --gpus all --ipc=host -p 127.0.0.1:8305:8000 \
  -v /srv/model-cache:/hf:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai:v0.29.0 \
  --model $MODEL --revision $REV --max-model-len 8192 --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 512 --no-enable-flashinfer-autotune "${eager[@]}" $OFFLOAD >/dev/null || exit 1
echo "o70 $ARM: $OFFLOAD ${eager[*]:-graphs}"
t0=$(date +%s)
until curl -sf $B/health >/dev/null; do
  if [ "$(docker inspect -f '{{.State.Running}}' $name 2>/dev/null)" != true ]; then
    echo "container exited before ready ($(( $(date +%s)-t0 ))s)"
    docker logs $name 2>&1 | grep -E "Error|error|memory" | tail -4 | cut -c1-260; exit 3; fi
  [ $(( $(date +%s)-t0 )) -gt 1500 ] && { echo "not ready in 1500s"; exit 3; }
  sleep 10
done
echo "ready in $(( $(date +%s)-t0 ))s"
docker logs $name 2>&1 | grep -E "Available KV cache memory|GPU KV cache size|Maximum concurrency|Model loading took|offloaded|Offloader" \
  | sed 's/^.*\] //' | cut -c1-200 | tee "$out/kv.txt"
{ nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,power.draw --format=csv; free -m; } > "$out/mem.txt"
for c in 1 4 16; do
  [ -f "$out/bench-c$c.json" ] && continue
  python3 /home/david/gpu-lab/bench/bench.py --base $B --model $MODEL --concurrency $c --repeats 2 \
    --label "o70 $ARM c=$c" --json-out "$out/bench-c$c.json" > "$out/bench-c$c.log" 2>&1
  echo "bench c=$c exit $?: $(grep -E 'decode rate|TTFT' "$out/bench-c$c.log" | tail -2 | tr -s ' ' | tr '\n' ' ')"
done
if [ "${EVAL:-0}" = 1 ]; then
  t=$(date +%s)
  TAG=o70-$ARM ARGS="--render llama31-meta-json" B=$B bash "$here/probes/l70_eval.sh" | tee "$out/eval.log"
  echo "eval total $(( $(date +%s)-t ))s" | tee -a "$out/eval.log"
fi

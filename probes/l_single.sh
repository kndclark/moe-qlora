#!/bin/bash
# Gap ledger L3 / L4 / L5 / L11 (plan.md "Final stage"): base Lightning NVFP4 on ONE card,
# the gate server's flags (g6_eval.sh, without LoRA) plus EXTRA, then gpu-lab bench.py
# decode at c=1 and c=16 (as G7c2) and v1 thinking off (as G7c2) against it.
# NODE=laptop serves on 127.0.0.1:8303 (sm_120); NODE=desktop on lab-desktop:8303 (sm_86,
# needs EXTRA to include --linear-backend marlin). EXTRA is split on spaces, so JSON in it
# must have none, e.g. --speculative-config {"method":"mtp","num_speculative_tokens":3}.
# Outputs: results/lsingle/<LABEL>/{serve.log,bench-c1.json,bench-c16.json,
#   research-eval-v1-lightning-nothink-<LABEL>.{json,log}}
# usage: NODE=laptop|desktop LABEL=name [EXTRA="..."] [L_EAGER=""] l_single.sh
#   L_EAGER="" drops --enforce-eager (L2b: CUDA graphs); unset keeps it.
set -u
NODE=${NODE:?laptop or desktop} LABEL=${LABEL:?names the run}
here=$(cd "$(dirname "$0")/.." && pwd)
out=$here/results/lsingle/$LABEL
mkdir -p "$out"
M=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
name=lsingle-$LABEL
if [ "$NODE" = desktop ]; then
  B=http://lab-desktop:8303 port=lab-desktop:8303:8000 run=(ssh llm sudo docker)
else
  B=http://127.0.0.1:8303 port=127.0.0.1:8303:8000 run=(docker)
fi
python3 "$here/probes/g7a_eval.py" --selfcheck | tail -1 | grep -qx "selfcheck PASS" || { echo "g7a_eval.py --selfcheck failed"; exit 4; }
stop() {
  "${run[@]}" logs $name > "$out/serve.log" 2>&1 || true
  "${run[@]}" rm -f $name >/dev/null 2>&1
}
trap stop EXIT
"${run[@]}" run -d --name $name --gpus all --ipc=host -p $port \
  -v /srv/model-cache:/hf:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai:v0.29.0 \
  --model $M --revision bee7596271d1495f6992ae224aefde4410e816b8 --served-model-name lightning-nvfp4 \
  --kv-cache-dtype fp8 --mamba-cache-mode align --moe-backend marlin \
  --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.85 ${L_EAGER---enforce-eager} \
  ${EXTRA:-} >/dev/null || exit 1
echo "$NODE $LABEL extra: ${EXTRA:-none}"
t0=$(date +%s)
until curl -sf $B/health >/dev/null; do
  if [ "$("${run[@]}" inspect -f '{{.State.Running}}' $name 2>/dev/null)" != true ]; then
    echo "container exited before ready ($(( $(date +%s)-t0 ))s)"
    "${run[@]}" logs $name 2>&1 | grep -E "Error|error" | tail -3 | cut -c1-240; exit 2; fi
  [ $(( $(date +%s)-t0 )) -gt 900 ] && { echo "not ready in 900s"; exit 3; }
  sleep 10
done
echo "ready in $(( $(date +%s)-t0 ))s"
"${run[@]}" logs $name 2>&1 | grep -E "Available KV cache memory|GPU KV cache size" | sed 's/^.*\] //' | cut -c1-140
for c in 1 16; do
  python3 /home/david/gpu-lab/bench/bench.py --base $B --model lightning-nvfp4 --concurrency $c --repeats 2 \
    --label "$LABEL c=$c" --json-out "$out/bench-c$c.json" > "$out/bench-c$c.log" 2>&1
  echo "bench c=$c exit $?: $(grep -E 'decode rate' "$out/bench-c$c.log" | tail -1 | tr -s ' ')"
done
label=v1-lightning-nothink-$LABEL
python3 "$here/probes/g7a_eval.py" --base $B --model lightning-nvfp4 --label "$label" --set v1 \
  --max-calls 3 --temperature 0 --window 4000 --seed 20260923 --concurrency 16 --thinking off --max-tokens 512 \
  --out "$out/research-eval-$label.json" > "$out/research-eval-$label.log" 2>&1
echo "v1 eval exit $?"
"${run[@]}" logs $name 2>&1 | grep -iE "spec_decode|acceptance|draft" | tail -3 | sed 's/^.*\] //' | cut -c1-200

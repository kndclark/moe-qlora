#!/bin/bash
# G7c (plan.md "G7c"): base Lightning NVFP4 pooled across both cards (pipeline parallel 2)
# through gpu-lab's `lab pool up`, with G7b's flags (--linear-backend marlin for sm_86).
# Records the head's serve log (KV lines), base v1 thinking off through probes/g7a_eval.py
# (against base's laptop run, results/research-eval-v1-lightning-nothink-g6srv.json), and
# gpu-lab bench/bench.py decode at c=1 and c=16. Always ends with `lab pool down` and then
# `lab up` on the desktop, which pool down does not do (it leaves llama-swap stopped).
# usage: [POOL_GPU_UTIL=u] [G7C_OUT=dir] [G7C_MORE_FLAGS="..."] g7c_pool.sh
#   G7C_OUT names the folder under results/ (default g7c); G7C_MORE_FLAGS is appended
#   to the pool's vLLM flags (G7c2: "--max-num-seqs 16"). Unset = the original G7c run.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
out=$here/results/${G7C_OUT:-g7c}
mkdir -p "$out"
LAB=/home/david/gpu-lab/bin/lab
M=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
B=http://lab-desktop:8200
finish() {
  ssh llm "sudo docker exec ray-head cat /tmp/vllm-pool.log" > "$out/vllm-pool.log" 2>&1
  "$LAB" pool down > "$out/pool-down.log" 2>&1; echo "pool down exit $?"
  ssh llm "$LAB up" > "$out/desktop-lab-up.log" 2>&1
  echo "desktop lab up exit $?; llama-swap $(ssh llm systemctl is-active llama-swap)"
}
trap finish EXIT
POOL_MODEL=$M POOL_MAXLEN=16384 POOL_WAIT_SECS=900 \
POOL_EXTRA_FLAGS="--revision bee7596271d1495f6992ae224aefde4410e816b8 --kv-cache-dtype fp8 --mamba-cache-mode align --moe-backend marlin --linear-backend marlin --enforce-eager${G7C_MORE_FLAGS:+ $G7C_MORE_FLAGS}" \
  "$LAB" pool up > "$out/pool-up.log" 2>&1
rc=$?
echo "pool up exit $rc $(date +%T)"
if [ $rc -ne 0 ]; then
  ssh llm "sudo docker exec ray-head cat /tmp/vllm-pool.log" 2>/dev/null | grep -E "Error|error" | tail -4 | cut -c1-240
  exit 1
fi
ssh llm "sudo docker exec ray-head grep -E 'KV cache size|Maximum concurrency|Available KV' /tmp/vllm-pool.log" | sed 's/^.*INFO/INFO/' | cut -c1-200
python3 "$here/probes/g7a_eval.py" --base $B --model $M --label v1-lightning-nothink-pool --set v1 \
  --max-calls 3 --temperature 0 --window 4000 --seed 20260923 --concurrency 16 --thinking off --max-tokens 512 \
  --out "$out/research-eval-v1-lightning-nothink-pool.json" > "$out/research-eval-v1-lightning-nothink-pool.log" 2>&1
echo "v1 eval exit $?"
for c in 1 16; do
  python3 /home/david/gpu-lab/bench/bench.py --base $B --model $M --concurrency $c --repeats 2 \
    --label "pool Lightning NVFP4 c=$c" --json-out "$out/bench-c$c.json" > "$out/bench-c$c.log" 2>&1
  echo "bench c=$c exit $?"; tail -4 "$out/bench-c$c.log"
done

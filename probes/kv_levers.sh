#!/bin/bash
# KV levers (docs/laptop-memory-levers.md): G6q's eval server (probes/g6_eval.sh flags) plus
# --lora-target-modules ("tm"), which stops vLLM wrapping the 2,944 routed experts in rank-16
# LoRA slots the adapter never uses (0.82 GiB). Each phase starts one server with the extra
# flags in the table below, runs v1, v2, alert and trap3 thinking-on, and keeps its log.
# Outputs go to results/kv-levers/; kv_levers_agree.py compares them with G6q's three runs.
# usage: kv_levers.sh PHASE...   (or "round4" / "round5" for the doc's rows 2-9 / 10-14)
# Run with Chrome and Antigravity closed: vLLM charges their GPU memory to KV.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)   # inside a repo: v1 = 158 items
cd "$here" || exit 9
J=$here/results/kv-levers
mkdir -p "$J"
adapter=/home/david/moe-qlora/results/g6q-train-adapter
B=http://127.0.0.1:8303
name=kvlever-tm
TM=(--lora-target-modules q_proj k_proj v_proj o_proj in_proj up_proj down_proj)
LORA=(--enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules "g6q=/adapter")
G92=(--gpu-memory-utilization 0.92)
F16=(--mamba-ssm-cache-dtype float16)
NONE=(--mamba-cache-mode none --no-enable-prefix-caching)
python3 "$here/probes/g7a_eval.py" --selfcheck | tail -1 | grep -qx "selfcheck PASS" || { echo "selfcheck failed"; exit 4; }
restore() {
  docker rm -f "$name" >/dev/null 2>&1
  echo performance | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null
  echo "profile: $(cat /sys/firmware/acpi/platform_profile)"
}
trap restore EXIT
echo max-power | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null
common=(--max-calls 3 --temperature 0 --window 4000 --seed 20260923 --concurrency 16 --thinking on --max-tokens 4096)
phase() {  # tag [extra serve flags...]; env NOLORA=1 (no adapter), CAP=1 (capacity only), ESTCG=0
  local tag=$1; shift
  local extra=(); [ "${NOLORA:-0}" = 1 ] || extra=("${LORA[@]}" "${TM[@]}")
  local envx=(); [ -z "${ESTCG:-}" ] || envx=(-e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="$ESTCG")  # 0: no graph-memory estimate
  echo "== phase $tag $(date -u +%H:%M:%SZ)${NOLORA:+ nolora}$( [ "${CAP:-0}" = 1 ] && echo ' cap-only')"
  docker run -d --name "$name" --gpus all --ipc=host -p 127.0.0.1:8303:8000 \
    -v /srv/model-cache:/hf:ro -v "$adapter":/adapter:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 "${envx[@]}" \
    --pull never vllm/vllm-openai:v0.29.0 \
    --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --revision bee7596271d1495f6992ae224aefde4410e816b8 --served-model-name lightning-nvfp4 \
    --kv-cache-dtype fp8 --mamba-cache-mode align --moe-backend marlin \
    --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.85 \
    "${extra[@]}" "$@" >/dev/null || return 1
  local t0=$(date +%s)
  until curl -sf $B/health >/dev/null; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
      echo "container exited before ready ($(( $(date +%s)-t0 ))s)"; docker logs "$name" > "$J/serve-$tag.log" 2>&1
      docker rm -f "$name" >/dev/null 2>&1; return 2; fi
    [ $(( $(date +%s)-t0 )) -gt 900 ] && { echo "not ready in 900s"; docker logs "$name" > "$J/serve-$tag.log" 2>&1; docker rm -f "$name" >/dev/null 2>&1; return 3; }
    sleep 5
  done
  echo "ready in $(( $(date +%s)-t0 ))s; $(docker logs "$name" 2>&1 | grep -oE "Model loading took [0-9.]+ GiB|Available KV cache memory: [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens" | tr '\n' ';')"
  echo "  mem: $(docker logs "$name" 2>&1 | grep -oE "Total CPU offloaded parameters: [0-9.]+ GiB|Actual usage is [0-9.]+ GiB for consumed memory \(weights \+ non-torch\), [0-9.]+ GiB for peak activation, and [0-9.]+ GiB for CUDAGraph|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" | tr '\n' ';')"
  if [ "${CAP:-0}" = 1 ]; then
    local m=g6q; [ "${NOLORA:-0}" = 1 ] && m=lightning-nvfp4
    echo "  smoke: $(curl -s $B/v1/completions -H 'Content-Type: application/json' -d "{\"model\":\"$m\",\"prompt\":\"2+2=\",\"max_tokens\":4,\"temperature\":0}" | python3 -c 'import json,sys; print(repr(json.load(sys.stdin)["choices"][0]["text"]))' 2>&1)"
    docker logs "$name" > "$J/serve-$tag.log" 2>&1; docker rm -f "$name" >/dev/null 2>&1; return 0
  fi
  for s in v1 v2 alert trap3; do
    local t=$(date +%s)
    python3 "$here/probes/g7a_eval.py" --base $B --model g6q --label "$s-g6q-$tag" --set "$s" \
      "${common[@]}" --out "$J/research-eval-$s-g6q-$tag.json" > "$J/research-eval-$s-g6q-$tag.log" 2>&1
    echo "  $s-g6q-$tag: exit $?, $(( $(date +%s)-t ))s"
  done
  docker logs "$name" > "$J/serve-$tag.log" 2>&1
  echo "  max running/waiting: $(grep -oE "Running: [0-9]+ reqs, Waiting: [0-9]+ reqs" "$J/serve-$tag.log" | sort -t' ' -k2,2n -k5,5n | tail -1)"
  docker rm -f "$name" >/dev/null 2>&1
}
run() {  # the doc's lever table, row by row
  case $1 in
    tm-eager)         phase "$1" --enforce-eager ;;                                   # row 2
    tm-graphs)        phase "$1" ;;                                                   # row 3: fails
    tm-graphs092)     phase "$1" "${G92[@]}" ;;                                       # row 4
    tm-g092-f16|tm-g092-f16-r2|tm-g092-f16-r3)
                      phase "$1" "${G92[@]}" "${F16[@]}" ;;                           # row 5
    tm-g092-none)     phase "$1" "${G92[@]}" "${NONE[@]}" ;;                          # row 6
    tm-g092-f16-none) phase "$1" "${G92[@]}" "${F16[@]}" "${NONE[@]}" ;;              # row 7
    tm-g092-b512)     phase "$1" "${G92[@]}" --max-num-batched-tokens 512 ;;          # row 8
    tm-g085-b512)     phase "$1" --max-num-batched-tokens 512 ;;                      # row 9: fails
    tm-e092)          CAP=1 phase "$1" "${G92[@]}" --enforce-eager ;;                 # row 10
    nolora-g092)      NOLORA=1 CAP=1 phase "$1" "${G92[@]}" ;;                        # row 11
    tm-stack)         phase "$1" "${F16[@]}" "${NONE[@]}" --kv-cache-memory 2684354560 ;;  # row 12: 2.5 GiB
    tm-noest)         ESTCG=0 phase "$1" "${G92[@]}" "${F16[@]}" "${NONE[@]}" ;;      # row 13
    tm-off4)          phase "$1" "${G92[@]}" --cpu-offload-gb 4 --cpu-offload-params experts ;;  # row 14
    *) echo "unknown phase $1"; return 9 ;;
  esac
}
for p in "$@"; do
  case $p in
    round4) for q in tm-eager tm-graphs tm-graphs092 tm-g092-f16 tm-g092-b512 tm-g085-b512 \
                     tm-g092-none tm-g092-f16-none tm-g092-f16-r2 tm-g092-f16-r3; do run $q; done ;;
    round5) for q in nolora-g092 tm-e092 tm-stack tm-noest tm-off4; do run $q; done ;;
    *) run "$p" ;;
  esac
done
echo "== done $(date -u +%H:%M:%SZ)"

#!/bin/bash
# KV levers (docs/laptop-memory-levers.md): G6q's eval server (probes/g6_eval.sh flags) plus
# --lora-target-modules ("tm"), which stops vLLM wrapping the 2,944 routed experts in rank-16
# LoRA slots the adapter never uses (0.82 GiB). Each phase starts one server with the extra
# flags in the table below, runs v1, v2, alert and trap3 thinking-on, and keeps its log.
# Outputs go to results/kv-levers/; kv_levers_agree.py compares them with G6q's three runs.
# usage: kv_levers.sh PHASE...   (or "round4" / "round5" for the doc's rows 2-9 / 10-14,
#        "round6" / "round8" / "round7" / "round9" / "k6" / "p13" for the experts-in-RAM program's passes)
# Env per phase: MAXLEN / SEQS (default 16384 / 16); WORK="MODE..." runs probes/offload_bench.py
# modes (decode prefill agent needle experts) with BARGS instead of the eval, to bench-TAG.json.
# KS="KSTAGE=... VAR=..." loads the kstage plugin (probes/kstage) with those variables set;
# routing profiles from expert_cache_sim.py --emit-profile are mounted at /prof.
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
UVA=(--cpu-offload-params experts --cpu-offload-gb)   # + GiB
PREF="--lens 8192,32768,98304"
# KV per request, fit to vLLM's concurrency figures at 16k and 131k: 0.096 GiB of Mamba
# state plus 3,142 B per token. Row 4 has 1.69 GiB, row 14 (4 GiB of experts in RAM) 5.63.
AGENT="--agents 4 --start 8192 --chunk 4096 --gen 256 --target 98304"     # 1.53 GiB: fits row 4
AGENT4L="--agents 4 --start 8192 --chunk 4096 --gen 256 --target 126976"  # 1.87 GiB: does not
KC6="KSTAGE=cache KSTAGE_COLD_GB=4 KSTAGE_PROFILE=/prof/p7-all.json KSTAGE_STATS=30"
AGENT6="--agents 6 --start 8192 --chunk 4096 --gen 256 --target 98304"    # 2.30 GiB: does not
AGENT8L="--agents 8 --start 8192 --chunk 4096 --gen 256 --target 126976"  # 3.74 GiB: only with experts in RAM
cbest() {  # the fastest KSTAGE_COPY at 64 slots and 16 streams in x-cbench (82, one per SM, if none)
  local n; n=$(awk '/^slots 64 C=16 few/ { for (i = 1; i <= NF; i++) if ($i == "trace") t = $(i + 1) + 0
             n = $4; sub("few", "", n); sub(":", "", n); if (b == "" || t < b) { b = t; bn = n } }
            END { print bn }' "$J/cbench-laptop.txt" 2>/dev/null)
  echo "${n:-82}"
}
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
  if [ -n "${KS:-}" ]; then
    envx+=(-v "$here/probes/kstage":/k:ro -v "$J/profiles":/prof:ro -e PYTHONPATH=/k)
    for v in $KS; do envx+=(-e "$v"); done
  fi
  for v in ${ENVS:-}; do envx+=(-e "$v"); done   # extra container env, VAR=value ...
  echo "== phase $tag $(date -u +%H:%M:%SZ)${NOLORA:+ nolora}$( [ "${CAP:-0}" = 1 ] && echo ' cap-only')${MAXLEN:+ maxlen $MAXLEN}${WORK:+ work: $WORK}${KS:+ ks: $KS}"
  local ram0=$(free -m | awk '/^Mem:/{print $3}')
  docker run -d --name "$name" --gpus all --ipc=host -p 127.0.0.1:8303:8000 \
    -v /srv/model-cache:/hf:ro -v "$adapter":/adapter:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 "${envx[@]}" \
    --pull never vllm/vllm-openai:v0.29.0 \
    --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --revision bee7596271d1495f6992ae224aefde4410e816b8 --served-model-name lightning-nvfp4 \
    --kv-cache-dtype fp8 --mamba-cache-mode align --moe-backend marlin \
    --max-model-len "${MAXLEN:-16384}" --max-num-seqs "${SEQS:-16}" --gpu-memory-utilization 0.85 \
    "${extra[@]}" ${WORK:+--enable-prompt-tokens-details} "$@" >/dev/null || return 1
  local t0=$(date +%s)
  until curl -sf $B/health >/dev/null; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
      echo "container exited before ready ($(( $(date +%s)-t0 ))s)"; docker logs "$name" > "$J/serve-$tag.log" 2>&1
      docker rm -f "$name" >/dev/null 2>&1; return 2; fi
    [ $(( $(date +%s)-t0 )) -gt 900 ] && { echo "not ready in 900s"; docker logs "$name" > "$J/serve-$tag.log" 2>&1; docker rm -f "$name" >/dev/null 2>&1; return 3; }
    sleep 5
  done
  echo "ready in $(( $(date +%s)-t0 ))s; $(docker logs "$name" 2>&1 | grep -oE "Model loading took [0-9.]+ GiB|Available KV cache memory: [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens" | tr '\n' ';')"
  echo "  mem: $(docker logs "$name" 2>&1 | grep -oE "Total CPU offloaded parameters: [0-9.]+|Total GPU memory saved: [0-9.]+ GB, Static buffer pool: [0-9.]+ GB|Actual usage is [0-9.]+ GiB for consumed memory \(weights \+ non-torch\), [0-9.]+ GiB for peak activation, and [0-9.]+ GiB for CUDAGraph|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" | tr '\n' ';')"
  [ -z "${KS:-}" ] || docker logs "$name" 2>&1 | grep -E "kstage: (gather|hotcold|cache|installed|registered|prefetch|predict)" | sed 's/^.*kstage:/  kstage:/' | sort -u
  echo "  host RAM used: +$(( $(free -m | awk '/^Mem:/{print $3}') - ram0 )) MiB, $(free -m | awk '/^Mem:/{print $7}') MiB available"
  if [ "${CAP:-0}" = 1 ]; then
    local m=g6q; [ "${NOLORA:-0}" = 1 ] && m=lightning-nvfp4
    echo "  smoke: $(curl -s $B/v1/completions -H 'Content-Type: application/json' -d "{\"model\":\"$m\",\"prompt\":\"2+2=\",\"max_tokens\":4,\"temperature\":0}" | python3 -c 'import json,sys; print(repr(json.load(sys.stdin)["choices"][0]["text"]))' 2>&1)"
    docker logs "$name" > "$J/serve-$tag.log" 2>&1; docker rm -f "$name" >/dev/null 2>&1; return 0
  fi
  if [ -n "${WORK:-}" ]; then
    local t=$(date +%s)
    # shellcheck disable=SC2086  # WORK and BARGS are word lists
    python3 "$here/probes/offload_bench.py" --base $B --model g6q --out "$J/bench-$tag.json" $WORK ${BARGS:-} \
      > "$J/bench-$tag.log" 2>&1
    echo "  bench: exit $?, $(( $(date +%s)-t ))s"; grep -E "^  (decode|prefill|agent|needle|experts)" "$J/bench-$tag.log"
    docker logs "$name" > "$J/serve-$tag.log" 2>&1
    echo "  max running/waiting: $(grep -oE "Running: [0-9]+ reqs, Waiting: [0-9]+ reqs" "$J/serve-$tag.log" | sort -t' ' -k2,2n -k5,5n | tail -1)"
    docker rm -f "$name" >/dev/null 2>&1; return 0
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
    # experts-in-RAM program (doc section "Experts in RAM"): bench workloads, not the eval.
    # A/B: row 4 vs row 14 at a 128k context; the agent load (4 x 98k) fits both: cost side.
    p6-base|p6-off4)
      local o=(); [ "$1" = p6-off4 ] && o=("${UVA[@]}" 4)
      MAXLEN=131072 WORK="decode prefill agent" BARGS="$PREF $AGENT" phase "$1" "${G92[@]}" "${o[@]}" ;;
    p6-off2|p6-off8|p6-off12|p6-off16)   # C: size sweep; 16 = every routed expert (15.4 GiB)
      MAXLEN=131072 WORK=decode BARGS="--conc 1,4,16" phase "$1" "${G92[@]}" "${UVA[@]}" "${1#p6-off}" ;;
    p7-off4-b8k|p7-off4-b16k|p7-base-b8k|p7-base-b16k)   # D: prefill chunk under offload
      local o=(); case $1 in *off4*) o=("${UVA[@]}" 4) ;; esac
      local b=8192; case $1 in *b16k) b=16384 ;; esac
      MAXLEN=131072 WORK="decode prefill" BARGS="--conc 1,16 $PREF" phase "$1" "${G92[@]}" "${o[@]}" --max-num-batched-tokens $b ;;
    p7-pf-g4s1|p7-pf-g4s2|p7-pf-g2s2|p7-pf-all)   # E: prefetch whole layers; MoE layers: g4 7, g2 13, all 23
      local g=4 s=1; case $1 in *g4s2) s=2 ;; *g2s2) g=2 s=2 ;; *all) g=1 s=2 ;; esac
      MAXLEN=131072 WORK="decode prefill" BARGS="--conc 1,4,16 $PREF" phase "$1" "${G92[@]}" --offload-backend prefetch \
        --offload-group-size $g --offload-num-in-group 1 --offload-prefetch-step $s --offload-params experts ;;
    p7-off4-eager)    MAXLEN=131072 WORK=decode BARGS="--conc 1,4,16" phase "$1" "${G92[@]}" "${UVA[@]}" 4 --enforce-eager ;;  # H
    p7-off4-f16|p7-off4-f16-none)   # F: offload stacked with the state levers (align kept, then none)
      local n=(); [ "$1" = p7-off4-f16-none ] && n=("${NONE[@]}")
      ESTCG=0 MAXLEN=131072 WORK="decode agent" BARGS="--conc 1,16 $AGENT4L" phase "$1" "${G92[@]}" "${F16[@]}" "${n[@]}" "${UVA[@]}" 4 ;;
    p7-kvoff16)       # I: native KV offload to RAM (re-prefill avoided after eviction?)
      MAXLEN=131072 WORK=agent BARGS="$AGENT4L" phase "$1" "${G92[@]}" --kv-offloading-size 16 --kv-offloading-backend native ;;
    x-vmm)            echo "== phase $1 $(date -u +%H:%M:%SZ) (no server)"   # K: one tensor, device + host pages
                      docker run --rm --pull never --gpus all -v "$here/probes":/p:ro --entrypoint python3 \
                        vllm/vllm-openai:v0.29.0 /p/vmm_mixed_test.py 1024 2>&1 | grep -v '^$' | tee "$J/vmm-laptop.txt" ;;
    p7-experts)       WORK=experts phase "$1" "${G92[@]}" --enable-return-routed-experts ;;   # J: routing histogram
    p7-1m-pfall)      MAXLEN=1048576 WORK=needle BARGS="--lens 1040000" phase "$1" "${G92[@]}" --offload-backend prefetch \
                        --offload-group-size 1 --offload-num-in-group 1 --offload-prefetch-step 2 --offload-params experts ;;
    # B2: agent loads that overflow row 4's KV, with and without 4 GiB of experts in RAM
    p8-base-a4l|p8-off4-a4l|p8-base-a6|p8-off4-a6)
      local o=(); case $1 in *off4*) o=("${UVA[@]}" 4) ;; esac
      local w=$AGENT4L; case $1 in *a6) w=$AGENT6 ;; esac
      MAXLEN=131072 WORK=agent BARGS="$w" phase "$1" "${G92[@]}" "${o[@]}" ;;
    # G: full context. Row 4 holds one ~545k-token request; 1,048,576 tokens need 3.16 GiB.
    p8-512k-base|p8-512k-f16)   # recall at length, fp32 vs fp16 Mamba state, no offload
      local f=(); [ "$1" = p8-512k-f16 ] && f=("${F16[@]}")
      MAXLEN=524288 WORK=needle BARGS="--lens 32768,131072,262144,516000" phase "$1" "${G92[@]}" "${f[@]}" ;;
    p8-1m-base)       MAXLEN=1048576 CAP=1 phase "$1" "${G92[@]}" ;;   # predicted not to start
    p8-1m-off2|p8-1m-off4)
      MAXLEN=1048576 WORK=needle BARGS="--lens 1040000" phase "$1" "${G92[@]}" "${UVA[@]}" "${1#p8-1m-off}" ;;
    # K: the kstage plugin stages experts itself instead of Marlin reading host memory.
    # Desktop, 4 GiB in RAM: gather 1.6-1.7x UVA's decode, 2.1x its prefill, same greedy ids.
    x-copy|x-gsweep)  echo "== phase $1 $(date -u +%H:%M:%SZ) (no server)"   # host-read rates
                      local s=copy_bench.py; [ "$1" = x-gsweep ] && s=gather_sweep.py
                      docker run --rm --pull never --gpus all -v "$here/probes/kstage":/k:ro --entrypoint python3 \
                        vllm/vllm-openai:v0.29.0 /k/$s 2>&1 | grep -v '^$' | tee "$J/${1#x-}-laptop.txt" ;;
    x-ks-lora)   # greedy ids through G6q (the routed experts run inside vLLM's fused-MoE LoRA wrapper)
      echo "== phase $1 $(date -u +%H:%M:%SZ) (kstage_check.sh, port 8313)"
      local c; for c in "base|" "off4|" "off4-gather|KSTAGE=gather" "off4-gdma|KSTAGE=gather KSTAGE_DMA_M=256" \
                      "hotcold4|KSTAGE=hotcold KSTAGE_COLD_GB=4 KSTAGE_PROFILE=/prof/p7-all.json"; do
        local u=(); case ${c%%|*} in off4*) u=("${UVA[@]}" 4) ;; esac
        KC_DOCKER="-v $adapter:/adapter:ro -v $J/profiles:/prof:ro" KC_MODEL=g6q KC_LINEAR=auto \
          bash "$here/probes/kstage/kstage_check.sh" "lora-${c%%|*}" "${c#*|}" "${u[@]}" "${G92[@]}" "${LORA[@]}" "${TM[@]}"
      done ;;
    p9-ks-gather4|p9-ks-gather8|p9-ks-gather16)   # K2 size sweep (compare p6-off4/8/16)
      KS="KSTAGE=gather" MAXLEN=131072 WORK="decode prefill" BARGS="--conc 1,4,16 $PREF" \
        phase "$1" "${G92[@]}" "${UVA[@]}" "${1#p9-ks-gather}" ;;
    p9-ks-gather4-dma)   # batches of 256+ tokens copy each layer whole on the copy engine
      KS="KSTAGE=gather KSTAGE_DMA_M=256" MAXLEN=131072 WORK="decode prefill" BARGS="--conc 1,16 $PREF" \
        phase "$1" "${G92[@]}" "${UVA[@]}" 4 ;;
    p9-ks-gather4-a6)    # the agent load that overflows row 4 (compare p8-base-a6, p8-off4-a6)
      KS="KSTAGE=gather" MAXLEN=131072 WORK=agent BARGS="$AGENT6" phase "$1" "${G92[@]}" "${UVA[@]}" 4 ;;
    p9-ks-hotcold4|p9-ks-hotcold8)   # K5: hot experts on the card, cold in host pages, by profile
      KS="KSTAGE=hotcold KSTAGE_COLD_GB=${1#p9-ks-hotcold} KSTAGE_PROFILE=/prof/p7-all.json" MAXLEN=131072 \
        WORK="decode prefill" BARGS="--conc 1,4,16 $PREF" phase "$1" "${G92[@]}" ;;
    p9-pf-a6)            # prefetch's whole-layer copies on the agent load
      MAXLEN=131072 WORK=agent BARGS="$AGENT6" phase "$1" "${G92[@]}" --offload-backend prefetch \
        --offload-group-size 4 --offload-num-in-group 1 --offload-prefetch-step 1 --offload-params experts ;;
    p9-ks-dma32-s64)   # laptop: whole-layer DMA (51.6 GB/s) beats gather (36.5) from ~26 tokens a step
      KS="KSTAGE=gather KSTAGE_DMA_M=32" SEQS=64 MAXLEN=131072 WORK=decode BARGS="--conc 16,32,64" \
        phase "$1" "${G92[@]}" "${UVA[@]}" 4 ;;
    p9-ks-gather4-s64|p9-base-s64)   # more sequences: per-step expert reads saturate near c=64
      local o=(); [ "$1" = p9-ks-gather4-s64 ] && o=("${UVA[@]}" 4)
      local k=; [ "$1" = p9-ks-gather4-s64 ] && k="KSTAGE=gather"
      KS=$k SEQS=64 MAXLEN=131072 WORK=decode BARGS="--conc 16,32,64" phase "$1" "${G92[@]}" "${o[@]}" ;;
    p9-ks-gather4-async|p9-off4-async)   # CPU scheduling overlapped with the GPU step
      local k=; [ "$1" = p9-ks-gather4-async ] && k="KSTAGE=gather"
      KS=$k MAXLEN=131072 WORK=decode BARGS="--conc 1,4,16" phase "$1" "${G92[@]}" "${UVA[@]}" 4 --async-scheduling ;;
    p9-ks-gather4-eval)  KS="KSTAGE=gather" phase "$1" "${G92[@]}" "${UVA[@]}" 4 ;;   # quality, vs tm-off4
    # K6: hotcold's device rows, part of them LRU slots refilled from host homes on a miss.
    # Same KV as p9-ks-hotcold4; host RAM pays a home for every unpinned expert (all: 16.5 GB).
    p10-ks-cache4-s16|p10-ks-cache4-s64|p10-ks-cache4-all|p10-ks-cache4-all-freeze)
      local s=${1#p10-ks-cache4-}; local f=; [ "$s" = all-freeze ] && { s=all; f=" KSTAGE_FREEZE_M=256"; }
      KS="$KC6 KSTAGE_SLOTS=${s#s}$f" MAXLEN=131072 WORK="decode prefill" BARGS="--conc 1,4,16 $PREF" \
        phase "$1" "${G92[@]}" ;;
    p11-off4-fdo|p11-ks-cache4-fdo|p11-base-fdo|p11-off4-brk)   # T: decode graphs, no torch.compile
      local c='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' e= a=("${UVA[@]}" 4) k=
      case $1 in
        *-brk) c='{"mode":0}'; e=VLLM_USE_BREAKABLE_CUDAGRAPH=1 ;;   # piecewise graphs without compile
        *-ks-*) a=(); k="$KC6 KSTAGE_SLOTS=all" ;;
        *-base-*) a=() ;;
      esac
      KS="$k" ENVS="$e" MAXLEN=131072 WORK="decode prefill" BARGS="--conc 1,4,16 $PREF" \
        phase "$1" "${G92[@]}" "${a[@]}" --compilation-config "$c" ;;
    p10-ks-cache4-a6)    # the agent load (compare p9-ks-gather4-a6: 402 s, p8-off4-a6, p8-base-a6)
      KS="$KC6 KSTAGE_SLOTS=all" MAXLEN=131072 WORK=agent BARGS="$AGENT6" phase "$1" "${G92[@]}" ;;
    p10-ks-cache4-eval)  KS="$KC6 KSTAGE_SLOTS=all" phase "$1" "${G92[@]}" ;;   # quality, vs tm-off4
    # K6 evicting the expert with the lowest decayed use count (half-life ~44 steps), not the
    # least recent: the trace simulator gives it 19% less copy time than LRU on code at 16 streams.
    p12-ks-cache4-a6-lfu)
      KS="$KC6 KSTAGE_SLOTS=all KSTAGE_EVICT=lfu" MAXLEN=131072 WORK=agent BARGS="$AGENT6" phase "$1" "${G92[@]}" ;;
    p12-ks-cache4-all-lfu|p12-ks-cache4-s64-lfu)
      local s=${1#p12-ks-cache4-}; s=${s%-lfu}
      KS="$KC6 KSTAGE_SLOTS=${s#s} KSTAGE_EVICT=lfu" MAXLEN=131072 WORK="decode prefill" BARGS="--conc 1,4,16 $PREF" \
        phase "$1" "${G92[@]}" ;;
    # Prefetch again, now the race is known (vLLM 0.29.0: wrong logits when the offloaded layer
    # count n is not a multiple of the step). G4/K1 offloads n=7: step 2 needs KSTAGE_PFFIX.
    p10-pf-fix-g4s2|p10-pf-fix-a6)
      local w="decode prefill" b="--conc 1,4,16 $PREF"; [ "$1" = p10-pf-fix-a6 ] && { w=agent; b=$AGENT6; }
      KS="KSTAGE_PFFIX=1" MAXLEN=131072 WORK="$w" BARGS="$b" phase "$1" "${G92[@]}" --offload-backend prefetch \
        --offload-group-size 4 --offload-num-in-group 1 --offload-prefetch-step 2 --offload-params experts ;;
    p10-pf-g4k2s2)   # n=12: safe without the fix; 8.6 GB off the card
      MAXLEN=131072 WORK="decode prefill" BARGS="--conc 1,4,16 $PREF" phase "$1" "${G92[@]}" --offload-backend prefetch \
        --offload-group-size 4 --offload-num-in-group 2 --offload-prefetch-step 2 --offload-params experts ;;
    p10-off4-b12k)   # D: 16384 OOMed at startup
      MAXLEN=131072 WORK="decode prefill" BARGS="--conc 1,16 $PREF" phase "$1" "${G92[@]}" "${UVA[@]}" 4 \
        --max-num-batched-tokens 12288 ;;
    p10-ks-dma-b8k)   # K2 prefill: DMA copies each layer whole per batch, 4x fewer batches at 8k
      KS="KSTAGE=gather KSTAGE_DMA_M=256" MAXLEN=131072 WORK="decode prefill" BARGS="--conc 1,16 $PREF" \
        phase "$1" "${G92[@]}" "${UVA[@]}" 4 --max-num-batched-tokens 8192 ;;
    p10-ks-hotcold4-dma|p10-ks-hotcold4-dma-b8k)   # K5 prefill: the layer copied whole, then run
      local b=(); [ "$1" = p10-ks-hotcold4-dma-b8k ] && b=(--max-num-batched-tokens 8192)
      KS="KSTAGE=hotcold KSTAGE_COLD_GB=4 KSTAGE_PROFILE=/prof/p7-all.json KSTAGE_DMA_M=256" MAXLEN=131072 \
        WORK="decode prefill" BARGS="--conc 1,16 $PREF" phase "$1" "${G92[@]}" "${b[@]}" ;;
    p10-ks-predict-a2)   # O: next-layer routing prediction on agent text (experts stay on the card)
      KS="KSTAGE=predict KSTAGE_COLD_GB=4 KSTAGE_PROFILE=/prof/p7-all.json KSTAGE_STATS=60" MAXLEN=131072 \
        WORK=agent BARGS="--agents 2 --start 8192 --chunk 4096 --gen 256 --target 65536" phase "$1" "${G92[@]}" ;;
    # K6's copy kernel without a server: fixed cost per layer-step, one miss's copy, copies beside
    # compute; the default grid against KSTAGE_COPY=N (N programs walking the copy list).
    x-cbench)  echo "== phase $1 $(date -u +%H:%M:%SZ) (no server)"
               docker run --rm --pull never --gpus all -v "$here/probes":/p:ro -v "$J":/r:ro --entrypoint python3 \
                 vllm/vllm-openai:v0.29.0 -B /p/kstage/cache_bench.py /r/bench-p7-experts.json /r/profiles/p7-all.json \
                 --layers 4 --conc 1,4,16 --slots all,64,16 --few 32,82,164,328 2>&1 | grep -v Warning \
                 | tee "$J/cbench-laptop.txt" ;;
    p13-ks-cache4-all-cb|p13-ks-cache4-s64-cb)
      local s=${1#p13-ks-cache4-}; s=${s%-cb}; local n; n=$(cbest)
      KS="$KC6 KSTAGE_SLOTS=${s#s} KSTAGE_COPY=$n" MAXLEN=131072 WORK="decode prefill" BARGS="--conc 1,4,16 $PREF" \
        phase "$1" "${G92[@]}" ;;
    # More agents than the card's KV can run at once: 8 to 127k. Without offload 3 fit at full
    # length; native KV offload keeps the rest's prefixes in RAM (row I: 4 agents 519 -> 195 s) but
    # cannot run them; 4 GiB of experts in RAM (K6) runs all 8. And both together.
    p13-base-a8l|p13-kvoff8-a8l|p13-ks-cache4-a8l|p13-ks-cache4-kvoff8-a8l|p13-ks-cache4-a8l-cb)
      local o=() k=""
      case $1 in *kvoff8*) o=(--kv-offloading-size 8 --kv-offloading-backend native) ;; esac
      case $1 in *ks-cache4*) k="$KC6 KSTAGE_SLOTS=all" ;; esac
      case $1 in *-cb) k="$k KSTAGE_COPY=$(cbest)" ;; esac
      KS="$k" MAXLEN=131072 WORK=agent BARGS="$AGENT8L" phase "$1" "${G92[@]}" "${o[@]}" ;;
    *) echo "unknown phase $1"; return 9 ;;
  esac
}
for p in "$@"; do
  case $p in
    round4) for q in tm-eager tm-graphs tm-graphs092 tm-g092-f16 tm-g092-b512 tm-g085-b512 \
                     tm-g092-none tm-g092-f16-none tm-g092-f16-r2 tm-g092-f16-r3; do run $q; done ;;
    round5) for q in nolora-g092 tm-e092 tm-stack tm-noest tm-off4; do run $q; done ;;
    round6) for q in p6-base p6-off4 p6-off2 p6-off8 p6-off12 p6-off16; do run $q; done ;;
    round7) for q in p7-off4-b8k p7-off4-b16k p7-pf-g4s1 p7-pf-g4s2 p7-pf-g2s2 p7-pf-all \
                     p7-off4-eager p7-off4-f16 p7-off4-f16-none p7-kvoff16 p7-1m-pfall; do run $q; done ;;
    round8) for q in x-vmm p7-experts p8-base-a4l p8-off4-a4l p8-base-a6 p8-off4-a6 p8-1m-base p8-1m-off2 p8-1m-off4 \
                     p8-512k-base p8-512k-f16; do run $q; done ;;
    round9) for q in x-copy x-gsweep x-ks-lora p9-ks-gather4 p9-ks-gather4-dma p9-ks-gather4-a6 p9-ks-gather8 \
                     p9-ks-gather16 p9-ks-hotcold4 p9-ks-hotcold8 p9-pf-a6 p9-ks-gather4-s64 p9-base-s64 \
                     p9-ks-gather4-async p9-off4-async p9-ks-gather4-eval; do run $q; done ;;
    round10) for q in p10-pf-fix-g4s2 p10-pf-g4k2s2 p10-pf-fix-a6 p10-off4-b12k p10-ks-dma-b8k \
                      p10-ks-hotcold4-dma p10-ks-hotcold4-dma-b8k p10-ks-predict-a2; do run $q; done ;;
    k6) for q in p10-ks-cache4-s16 p10-ks-cache4-s64 p10-ks-cache4-all p10-ks-cache4-all-freeze \
                 p10-ks-cache4-a6 p10-ks-cache4-eval; do run $q; done ;;
    t) for q in p11-off4-fdo p11-ks-cache4-fdo p11-off4-brk p11-base-fdo; do run $q; done ;;
    lfu) for q in p12-ks-cache4-a6-lfu p12-ks-cache4-all-lfu p12-ks-cache4-s64-lfu; do run $q; done ;;
    p13) for q in x-cbench p13-base-a8l p13-kvoff8-a8l p13-ks-cache4-a8l p13-ks-cache4-kvoff8-a8l \
                  p13-ks-cache4-a8l-cb p13-ks-cache4-all-cb p13-ks-cache4-s64-cb; do run $q; done ;;
    *) run "$p" ;;
  esac
done
echo "== done $(date -u +%H:%M:%SZ)"

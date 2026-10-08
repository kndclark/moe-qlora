#!/bin/bash
# KV levers (docs/laptop-memory-levers.md): G6q's eval server (probes/g6_eval.sh flags) plus
# --lora-target-modules ("tm"), which stops vLLM wrapping the 2,944 routed experts in rank-16
# LoRA slots the adapter never uses (0.82 GiB). Each phase starts one server with the extra
# flags in the table below, runs v1, v2, alert and trap3 thinking-on, and keeps its log.
# Outputs go to results/kv-levers/; kv_levers_agree.py compares them with G6q's three runs.
# usage: kv_levers.sh PHASE...   (or "round4" / "round5" for the doc's rows 2-9 / 10-14,
#        "round6" / "round8" / "round7" / "round9" / "k6" / "p13" / "p14" / "p20" / "p21" / "p22" / "p23" / "p24" / "p25" / "p26" / "p27" / "p28" / "p29" / "p30" / "p31" / "p32" / "p33" / "p34" / "p35" / "p36" / "p37" / "p38" / "p39" / "p40" for the experts-in-RAM program's passes)
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
AGENT16L="--agents 16 --start 8192 --chunk 4096 --gen 256 --target 126976" # 7.49 GiB: spills K6's 5.71
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
  if [ "${TRACE:-0}" = 1 ]; then   # torch profiler traces (probes/kstage/trace_drive.py) to trace-TAG/
    mkdir -p "$J/trace-$tag"; envx+=(-v "$J/trace-$tag":/trace)
    set -- "$@" --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/trace \
      --profiler-config.torch_profiler_with_stack=false --profiler-config.ignore_frontend=true
  fi
  echo "== phase $tag $(date -u +%H:%M:%SZ)$( [ "${NOLORA:-0}" = 1 ] && echo ' nolora')$( [ "${CAP:-0}" = 1 ] && echo ' cap-only')${MAXLEN:+ maxlen $MAXLEN}${WORK:+ work: $WORK}${KS:+ ks: $KS}"
  # K6 pins its host pages with cuMemCreate in 2 MiB pieces; after days of uptime the page cache
  # left 34 MiB of free 2 MiB blocks and p19 died with CUDA_ERROR_OUT_OF_MEMORY at load. Drop the
  # clean cache and compact first (no-op without passwordless sudo).
  if sudo -n true 2>/dev/null; then
    sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'
  fi
  awk '/Normal/{for(i=14;i<=NF;i++)s+=$i*2^(i-5)/256} END{printf "  free in 2 MiB+ blocks: %.1f GiB\n", s/1024}' /proc/buddyinfo
  local ram0=$(free -m | awk '/^Mem:/{print $3}')
  docker run -d --name "$name" --gpus all --ipc=host -p 127.0.0.1:8303:8000 \
    -v /srv/model-cache:/hf:ro -v "$adapter":/adapter:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 "${envx[@]}" \
    --pull never vllm/vllm-openai:v0.29.0 \
    --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --revision bee7596271d1495f6992ae224aefde4410e816b8 --served-model-name lightning-nvfp4 \
    --kv-cache-dtype fp8 --mamba-cache-mode align --moe-backend "${MOE:-marlin}" \
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
  if [ "${TRACE:-0}" = 1 ] && [ -n "${WORK:-}" ]; then   # p39: profiler windows inside the bench
    local t=$(date +%s)
    # shellcheck disable=SC2086  # WORK and BARGS are word lists
    python3 "$here/probes/kstage/trace_agents.py" --server $B --at "${TAT:-15,170,320}" -- \
      --base $B --model g6q --out "$J/bench-$tag.json" $WORK ${BARGS:-} > "$J/bench-$tag.log" 2>&1
    echo "  bench: exit $?, $(( $(date +%s)-t ))s"; grep -E "^  (agent|window)" "$J/bench-$tag.log"
    ls "$J/trace-$tag" | sed 's/^/  trace file: /'
    docker logs "$name" > "$J/serve-$tag.log" 2>&1; docker rm -f "$name" >/dev/null 2>&1; return 0
  fi
  if [ "${TRACE:-0}" = 1 ]; then
    python3 "$here/probes/kstage/trace_drive.py" --base $B --model g6q --conc "${TCONC:-1,4,16}" > "$J/trace-$tag.log" 2>&1
    echo "  trace: exit $?"; sed 's/^/  /' "$J/trace-$tag.log"; ls "$J/trace-$tag" | sed 's/^/  trace file: /'
    docker logs "$name" > "$J/serve-$tag.log" 2>&1; docker rm -f "$name" >/dev/null 2>&1; return 0
  fi
  if [ -n "${WORK:-}" ]; then
    local t=$(date +%s) m=g6q; [ "${NOLORA:-0}" = 1 ] && m=lightning-nvfp4
    # shellcheck disable=SC2086  # WORK and BARGS are word lists
    python3 "$here/probes/offload_bench.py" --base $B --model $m --out "$J/bench-$tag.json" $WORK ${BARGS:-} \
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
    # Spill: 16 agents to 127k need 7.49 GiB of KV, more than K6's 5.71, so K6 alone must evict
    # or preempt; host KV (16 GiB: kvoff alone holds ~5.8 GiB off the card) should catch the spill.
    p14-kvoff16-a16l|p14-ks-cache4-a16l-cb|p14-ks-cache4-kvoff16-a16l-cb)
      local o=() k=""
      case $1 in *kvoff16*) o=(--kv-offloading-size 16 --kv-offloading-backend native) ;; esac
      case $1 in *ks-cache4*) k="$KC6 KSTAGE_SLOTS=all KSTAGE_COPY=$(cbest)" ;; esac
      KS="$k" MAXLEN=131072 WORK=agent BARGS="$AGENT16L" phase "$1" "${G92[@]}" "${o[@]}" ;;
    # P1 (row U): K6 with every batch of 64+ tokens outside CUDA graphs (prefill chunks) staged on
    # the copy engine, KSTAGE_DMA_BUF layers ahead, instead of Marlin reading ~33 cold experts a
    # layer over UVA. Compare p10-ks-cache4-all (98k 25.56 s), no offload 15.40, K2 DMA b8k 15.10;
    # agents vs p13-ks-cache4-a8l (589 s); quality vs p10-ks-cache4-eval.
    p15-ks-dma64-all|p15-ks-dma64-all-b8k|p15-ks-dma64-all-b1)
      local b=() k="$KC6 KSTAGE_SLOTS=all KSTAGE_DMA_M=64 KSTAGE_DMA_TIME=1"  # time: copy GB/s a chunk
      case $1 in *-b8k) b=(--max-num-batched-tokens 8192) ;; *-b1) k="$k KSTAGE_DMA_BUF=1" ;; esac
      KS="$k" MAXLEN=131072 WORK="decode prefill" BARGS="--conc 1,4,16 $PREF" phase "$1" "${G92[@]}" "${b[@]}" ;;
    p15-ks-dma64-a8l)  KS="$KC6 KSTAGE_SLOTS=all KSTAGE_DMA_M=64" MAXLEN=131072 WORK=agent BARGS="$AGENT8L" \
                         phase "$1" "${G92[@]}" ;;
    p15-ks-dma64-eval) KS="$KC6 KSTAGE_SLOTS=all KSTAGE_DMA_M=64" phase "$1" "${G92[@]}" ;;
    # 1M needle again: p8-1m-* never sent a prompt (the bench overshot to 1,198,435 tokens; it now
    # brackets the target). Then the same needle with K6 + P1 prefill.
    p15-1m-off4)       MAXLEN=1048576 WORK=needle BARGS="--lens 1040000" phase "$1" "${G92[@]}" "${UVA[@]}" 4 ;;
    p15-1m-ks-dma64)   KS="$KC6 KSTAGE_SLOTS=all KSTAGE_DMA_M=64" MAXLEN=1048576 WORK=needle BARGS="--lens 1040000" \
                         phase "$1" "${G92[@]}" ;;
    # P1 at 16 agents to 127k (7.49 GiB of KV against K6 + P1's 5.44 with one buffer): vs
    # p14-ks-cache4-a16l-cb (5,816 s), its kvoff16 twin (1,177 s) and host KV alone (744 s).
    p16-ks-dma64-b1-a16l-cb|p16-ks-dma64-b1-kvoff16-a16l-cb)
      local o=() k="$KC6 KSTAGE_SLOTS=all KSTAGE_DMA_M=64 KSTAGE_DMA_BUF=1 KSTAGE_COPY=$(cbest)"
      case $1 in *kvoff16*) o=(--kv-offloading-size 16 --kv-offloading-backend native) ;; esac
      KS="$k" MAXLEN=131072 WORK=agent BARGS="$AGENT16L" phase "$1" "${G92[@]}" "${o[@]}" ;;
    # P1 copies all 4 GiB for any step of KSTAGE_DMA_M+ tokens; p15's 80-208-token steps spent
    # 87-89 ms copying in a 99-112 ms span. A warm 98k prefill ends in a 208-token step: 128 and
    # 256 leave it to K6's own path. Compare p15-ks-dma64-all-b1 (warm 0.60 s, 8k warm 0.43).
    p16-ks-dma128-b1|p16-ks-dma256-b1)
      local m=${1#p16-ks-dma}; m=${m%-b1}
      KS="$KC6 KSTAGE_SLOTS=all KSTAGE_DMA_M=$m KSTAGE_DMA_BUF=1 KSTAGE_DMA_TIME=1" MAXLEN=131072 \
        WORK="decode prefill" BARGS="--conc 1,4,16 $PREF" phase "$1" "${G92[@]}" ;;
    # Row O: K6's misses at each concurrency (is a fill-ahead worth it at c=1?). Stats every second,
    # cumulative; layer-steps a second tell c=1 (~3,800), c=4 (~1,600) and c=16 (~500) apart.
    p17-ks-dma64-b1-miss)
      KS="${KC6/KSTAGE_STATS=30/KSTAGE_STATS=1} KSTAGE_SLOTS=all KSTAGE_DMA_M=64 KSTAGE_DMA_BUF=1" \
        MAXLEN=131072 WORK=decode BARGS="--conc 1,4,16" phase "$1" "${G92[@]}" ;;
    # Row O, fill-ahead (KSTAGE_AHEAD): at MoE layer p, layer p+1's gate on p's input picks its
    # likely experts; K6 assigns their slots and ce_helper copies the big rows on the copy engine
    # while the layers between run. vs p17 (same flags, no fill-ahead). The "ahead:" stats line
    # counts the predicted fills, "cache:" what the prediction missed. k10: top 10, not 6.
    p18-ks-dma64-b1-ahead|p18-ks-dma64-b1-ahead-k10)
      local k="${KC6/KSTAGE_STATS=30/KSTAGE_STATS=1} KSTAGE_SLOTS=all KSTAGE_DMA_M=64 KSTAGE_DMA_BUF=1 KSTAGE_AHEAD=1"
      case $1 in *-k10) k="$k KSTAGE_AHEAD_K=10" ;; esac
      KS="$k" MAXLEN=131072 WORK=decode BARGS="--conc 1,4,16" phase "$1" "${G92[@]}" ;;
    # Correctness (a fill that desynced from its layer would serve the wrong expert): vs
    # p15-ks-dma64-eval. Then 8 agents to 127k with and without it.
    p18-ks-dma64-b1-ahead-eval)
      KS="$KC6 KSTAGE_SLOTS=all KSTAGE_DMA_M=64 KSTAGE_DMA_BUF=1 KSTAGE_AHEAD=1" phase "$1" "${G92[@]}" ;;
    p18-ks-dma64-b1-a8l|p18-ks-dma64-b1-ahead-a8l)
      local k="$KC6 KSTAGE_SLOTS=all KSTAGE_DMA_M=64 KSTAGE_DMA_BUF=1"
      case $1 in *-ahead-*) k="$k KSTAGE_AHEAD=1" ;; esac
      KS="$k" MAXLEN=131072 WORK=agent BARGS="$AGENT8L" phase "$1" "${G92[@]}" ;;
    # Row O, lever B1: kernel traces of steady decode (1 / 4 / 16 streams), fill-ahead off and on,
    # so each node it adds to a layer has a GPU time; probes/kstage/trace_ahead.py compares them.
    p19-trace-base|p19-trace-ahead)
      local k="$KC6 KSTAGE_SLOTS=all KSTAGE_DMA_M=64 KSTAGE_DMA_BUF=1"
      case $1 in *-ahead) k="$k KSTAGE_AHEAD=1" ;; esac
      KS="$k" MAXLEN=131072 TRACE=1 phase "$1" "${G92[@]}" ;;
    # Rows K/O, the miss copy: p17-p19 ran the default grid, a program per 512 words of every
    # possible copy, 224 us a big row at 96 possible with none live (copy-rows-probe.txt); 1.5
    # with _copy_rows_live, 12 with KSTAGE_COPY=82. Tags name the levers: few (COPY=82), live
    # (COPY_LIVE=82), lfu, ahead; -rN is the draw, -trace a p19-style trace. p21 (B5): one
    # concurrency a server (-c1/-c4/-c16), since the cumulative stats lines carry no times; the
    # last "kstage: cache:" line of serve-TAG.log counts ahead fills used / evicted unused.
    # p24: fuse = KSTAGE_AHEAD_FUSE, the predictor in one launch (ahead_fused_test.py: 8-11 vs
    # 17-34 us a layer at 1-16 tokens), against live + LFU with and without the unfused ahead.
    # p25 (B6): where ahead's ~29 us a layer goes at one stream, fused or not (p24): dry =
    # KSTAGE_AHEAD_DRY, the predictor and flags with no copies; tick = KSTAGE_AHEAD_TICK, the flag
    # waits timed on the GPU ("ahead waits:" in serve-TAG.log).
    # p26: at one stream, dry with parts left out (-noX: KSTAGE_AHEAD_SKIP, X of pred, req, wait,
    # ah, helper) places the ~18 us a layer p25 left unplaced; at 16 streams, fewer ids predicted a
    # token (-kN: KSTAGE_AHEAD_K=N, 6 by default) cuts the fills, 43% of them evicted unused.
    # p27: p26's K=2 won at 16 streams in one draw; -rN repeats it. -minN = KSTAGE_AHEAD_MIN
    # (B4): graphs of fewer tokens skip ahead, so one stream pays none of its fixed cost.
    # p28: torch-profiler traces, live + LFU vs dry fused vs K=3 at 1/4/16 streams: is the fused
    # predictor's ~18 us a layer in the server (7.6 in isolation) its kernel or idle around it?
    # K=3 (the best at 16 streams in p26/p27): how much of its step is still fill waits?
    # p29: K=3 from 8 tokens (B4) on the loads it is for, 8 and 16 agents to 127k (-a8l, -a16l;
    # -kvoff16 adds prefix KV in RAM), against live + LFU in the same round (p22: 418 / 817 s).
    # p30: p28 found the fused predictor on the expert branch's path (16-22 us a layer in the
    # server); -side = KSTAGE_AHEAD_SIDE, on a side stream beside layer p's experts. -staleN =
    # KSTAGE_AHEAD_STALE=N: a fill evicts only experts under N uses' worth of LFU count (K=3's
    # evictions cause 0.74 misses a layer-step). -eval: answers, vs p15-ks-dma64-eval.
    p20-*|p21-*|p24-*|p25-*|p26-*|p27-*|p28-*|p29-*|p30-*|p32-*)
      local c=1,4,16
      case $1 in *-c16*) c=16 ;; *-c1*) c=1 ;; *-c4*) c=4 ;; *-c8*) c=8 ;; esac
      local k="${KC6/KSTAGE_STATS=30/KSTAGE_STATS=1} KSTAGE_SLOTS=all KSTAGE_DMA_M=64 KSTAGE_DMA_BUF=1"
      case $1 in *-few*) k="$k KSTAGE_COPY=82" ;; esac
      case $1 in *-live*) k="$k KSTAGE_COPY_LIVE=82" ;; esac
      case $1 in *-lfu*) k="$k KSTAGE_EVICT=lfu" ;; esac
      case $1 in *-ahead*) k="$k KSTAGE_AHEAD=1" ;; esac
      case $1 in *-fuse*) k="$k KSTAGE_AHEAD_FUSE=1" ;; esac
      case $1 in *-dry*) k="$k KSTAGE_AHEAD_DRY=1" ;; esac
      case $1 in *-tick*) k="$k KSTAGE_AHEAD_TICK=1" ;; esac
      case $1 in *-k[0-9]-*) local kk=${1#*-k}; k="$k KSTAGE_AHEAD_K=${kk%%-*}" ;; esac
      case $1 in *-min[0-9]*) local mm=${1#*-min}; k="$k KSTAGE_AHEAD_MIN=${mm%%-*}" ;; esac
      case $1 in *-side*) k="$k KSTAGE_AHEAD_SIDE=1" ;; esac
      case $1 in *-stale[0-9]*) local ss=${1#*-stale}; k="$k KSTAGE_AHEAD_STALE=${ss%%-*}" ;; esac
      local x s=
      for x in pred req wait ah helper; do case $1 in *-no$x-*) s="$s${s:+,}$x" ;; esac; done
      [ -z "$s" ] || k="$k KSTAGE_AHEAD_SKIP=$s"
      local o=() w=$AGENT8L
      case $1 in *-kvoff16*) o=(--kv-offloading-size 16 --kv-offloading-backend native) ;; esac
      case $1 in *-a16l) w=$AGENT16L ;; esac
      case $1 in
        *-trace) KS="${k/KSTAGE_STATS=1/KSTAGE_STATS=30}" MAXLEN=131072 TRACE=1 phase "$1" "${G92[@]}" ;;
        *-a8l|*-a16l) KS="$k" MAXLEN=131072 WORK=agent BARGS="$w" phase "$1" "${G92[@]}" "${o[@]}" ;;
        *-eval) KS="$k" phase "$1" "${G92[@]}" ;;
        *) KS="$k" MAXLEN=131072 WORK=decode BARGS="--conc $c" phase "$1" "${G92[@]}" ;;
      esac ;;
    # Rows B/K: p20's c=16 winner (live copy + LFU, 419 tok/s vs 343) on the long-context agent
    # loads. 8 x 127k vs p18-ks-dma64-b1-a8l (grid copy, LRU: 458 s) and prefix KV in RAM alone
    # (374 s); 16 x 127k with prefix KV in RAM vs p16 (COPY=82, LRU: 850 s) and that alone (744 s).
    # p23: the same with -cgN, N GiB of experts in RAM instead of 4. 8 x 127k runs 3.74 GiB of
    # KV against K6's 5.44, so ~1.5 GiB more experts could stay on the card.
    p22-*|p23-*)
      local o=() w=$AGENT8L k="$KC6 KSTAGE_SLOTS=all KSTAGE_DMA_M=64 KSTAGE_DMA_BUF=1 KSTAGE_COPY_LIVE=82"
      case $1 in *-cg*) local g=${1#*-cg}; k="${k/KSTAGE_COLD_GB=4/KSTAGE_COLD_GB=${g%%-*}}" ;; esac
      case $1 in *-lfu*) k="$k KSTAGE_EVICT=lfu" ;; esac
      case $1 in *kvoff16*) o=(--kv-offloading-size 16 --kv-offloading-backend native) ;; esac
      case $1 in *-a16l) w=$AGENT16L ;; esac
      KS="$k" MAXLEN=131072 WORK=agent BARGS="$w" phase "$1" "${G92[@]}" "${o[@]}" ;;
    # p31: the prefill chunk on the agent loads. To 127k the agents prefill ~1.4M uncached tokens,
    # and K6 stages its ~4 GiB of cold experts once a chunk of 64+ tokens (row U: 98k 18.0 s at
    # 2,048 tokens a step, 13.6 s at 8,192), so larger chunks should cut agent wall. Host KV
    # (-base: no K6, all experts on the card; p13 374 s) gets the same chunk. -bNk = N x 1024
    # tokens a step; -ahead = K=3 from 8 tokens; -c1 = one stream, p29's unresolved pair again.
    # p34: the host-KV-alone stall at 8k (p31), logged (-ujdiag), then with the waiting queues
    # rotated while nothing runs (-unjam, probes/kstage _unjam).
    # p35: -cgN (N GiB of experts in RAM, as p23) at 8k chunks, where p23's 3 GiB cut 2k's 418 s
    # to 389; and second draws of host KV alone at 4k chunks (p33: 351 s, 696 s).
    # p36: host KV alone at 6k chunks, unjammed: between p33's 4k (351 s, 696 s) and p34's 8k
    # (415 s, 835 s, 239k tokens of KV).
    # p37: the cold tier with host KV on. p35: 3 GiB + host KV at 8k ran 8 agents in 359 s (host
    # KV alone 351) and 16 in 716 s (697); without host KV, 768 s (8 x 127k overflowed 1.01M).
    # p38: the MoE kernel. Auto picks Marlin (FP4 unpacked to bf16) because any --enable-lora
    # rejects experts without LoRA support (modular_kernel.py), even when the experts are not
    # targets; -nolora drops the adapter, -moeX swaps in a native FP4 kernel for sm_120.
    # -exact keeps B12x's FP8 block scales exact; -kvbN pins KV at N/100 GiB (p38f: Marlin at
    # B12x's 1.05 GiB, to split B12x's agent result into kernel and KV).
    # p39 (row L): where the agent wall goes. -trace opens torch-profiler windows at 15, 170 and
    # 320 s into the bench (TAT; trace_agents.py), i.e. shallow, middle and deep contexts.
    # p40: what p39 found. Deep 4k prefill steps are 54-64% attention (FlashInfer fa2, bf16 Q over
    # fp8 KV: sm_120 has no fp8-Q prefill), so -kvbf16 drops the in-kernel fp8 conversion, -triton
    # and -fa2 swap the backend. And padded decode rows: Lightning's n_group=1 bias router stays
    # GroupedTopKRouter, which ignores VLLM_MOE_SKIP_PADDING, so rows padded up to a capture size
    # route stale tokens into real experts (cold misses under K6); -cs8 captures every size to 8,
    # at KV pinned by -kvb so the graph-memory estimate cannot move KV.
    p31-*|p33-*|p34-*|p35-*|p36-*|p37-*|p38-*|p39-*|p40-*|p41-*)
      local o=() w=$AGENT8L k="${KC6/KSTAGE_STATS=30/KSTAGE_STATS=1} KSTAGE_SLOTS=all KSTAGE_DMA_M=64"
      local nl=0 mb=marlin
      case $1 in *-nolora*) nl=1 ;; esac
      case $1 in *-moefic*) mb=flashinfer_cutlass ;; *-moeb12*) mb=flashinfer_b12x ;; *-moecut*) mb=cutlass ;; esac
      k="$k KSTAGE_DMA_BUF=1 KSTAGE_COPY_LIVE=82 KSTAGE_EVICT=lfu"
      case $1 in *-cg*) local g=${1#*-cg}; k="${k/KSTAGE_COLD_GB=4/KSTAGE_COLD_GB=${g%%-*}}" ;; esac
      case $1 in *-base*) k="" ;; esac
      case $1 in *-lgate*) k="$k KSTAGE_MOE_NOLORA=1" ;; esac   # native kernel with the adapter
      case $1 in *-moeb12*) k="$k KSTAGE_B12X_FIX=1" ;; esac   # padded intermediate (vllm_kstage.py)
      case $1 in *-exact*) k="$k KSTAGE_B12X_EXACT=1" ;; esac   # B12x keeps the FP8 block scales exact
      case $1 in *-ujdiag*) k="$k KSTAGE_UNJAM=diag" ;; *-unjam*) k="$k KSTAGE_UNJAM=1" ;; esac
      case $1 in *-ahead*) k="$k KSTAGE_AHEAD=1 KSTAGE_AHEAD_FUSE=1 KSTAGE_AHEAD_K=3 KSTAGE_AHEAD_MIN=8" ;; esac
      case $1 in
        *-kvoff8*) o=(--kv-offloading-size 8 --kv-offloading-backend native) ;;
        *-kvoff16*) o=(--kv-offloading-size 16 --kv-offloading-backend native) ;;
      esac
      case $1 in *-b[0-9]*k-*) local b=${1##*-b}; o+=(--max-num-batched-tokens $((${b%%k*} * 1024))) ;; esac
      case $1 in *-a16l) w=$AGENT16L ;; esac
      # bfloat16, not auto: auto takes the checkpoint's kv_cache_quant_algo, FP8 here (p40's kvbf16 ran fp8)
      case $1 in *-kvbf16*) o+=(--kv-cache-dtype bfloat16) ;; esac
      case $1 in *-triton*) o+=(--attention-backend TRITON_ATTN) ;; *-fa2*) o+=(--attention-backend FLASH_ATTN) ;; esac
      case $1 in *-cs8*) o+=(--compilation-config '{"cudagraph_capture_sizes": [1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32]}') ;; esac
      case $1 in *-kvb[0-9]*) local kb=${1##*-kvb}; o+=(--kv-cache-memory-bytes $((${kb%%-*} * 1073741824 / 100))) ;; esac
      case $1 in
        *-eval) NOLORA=$nl MOE=$mb KS="$k" MAXLEN=131072 phase "$1" "${G92[@]}" ;;   # quality, vs tm-graphs092
        *-c1-*) NOLORA=$nl MOE=$mb KS="$k" MAXLEN=131072 WORK=decode BARGS="--conc 1" phase "$1" "${G92[@]}" ;;
        *-pref*) NOLORA=$nl MOE=$mb KS="$k" MAXLEN=131072 WORK="decode prefill" BARGS="--conc 1,16 $PREF" \
                   phase "$1" "${G92[@]}" "${o[@]}" ;;
        *-trace-*) NOLORA=$nl MOE=$mb KS="$k" MAXLEN=131072 TRACE=1 WORK=agent BARGS="$w" phase "$1" "${G92[@]}" "${o[@]}" ;;
        *) NOLORA=$nl MOE=$mb KS="$k" MAXLEN=131072 WORK=agent BARGS="$w" phase "$1" "${G92[@]}" "${o[@]}" ;;
      esac ;;
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
    p14) for q in p14-kvoff16-a16l p14-ks-cache4-a16l-cb p14-ks-cache4-kvoff16-a16l-cb; do run $q; done ;;
    p15) for q in p15-ks-dma64-all p15-ks-dma64-all-b8k p15-ks-dma64-all-b1 p15-ks-dma64-a8l \
                  p15-1m-off4 p15-1m-ks-dma64 p15-ks-dma64-eval; do run $q; done ;;
    p16) for q in p16-ks-dma128-b1 p16-ks-dma256-b1 p16-ks-dma64-b1-a16l-cb \
                  p16-ks-dma64-b1-kvoff16-a16l-cb; do run $q; done ;;
    p17) for q in p17-ks-dma64-b1-miss; do run $q; done ;;
    p20) for q in base few live live-lfu live-ahead live-lfu-ahead; do run p20-$q-r1; done
         for q in live-lfu-ahead live-ahead live-lfu live few base; do run p20-$q-r2; done
         run p20-live-trace ;;
    p21) for c in 1 4 16; do for q in live-ahead live-lfu-ahead; do run p21-$q-c$c; done; done ;;
    p22) for q in live-lfu-a8l live-a8l live-lfu-kvoff16-a16l; do run p22-$q; done ;;
    p23) for g in 3 2.5; do run p23-cg$g-lfu-a8l; done ;;
    p24) for c in 1 4 16; do for q in live-lfu-ahead-fuse live-lfu live-lfu-ahead; do run p24-$q-c$c; done; done ;;
    p28) for q in live-lfu live-lfu-ahead-fuse-dry live-lfu-ahead-fuse-k3; do TCONC=1,4,16 run p28-$q-trace; done ;;
    p29) a=live-lfu-ahead-fuse-k3-min8
         for q in live-lfu $a; do run p29-$q-a8l; done
         for q in live-lfu $a; do run p29-$q-kvoff16-a16l; done
         for r in 1 2; do for q in live-lfu $a; do run p29-$q-c1-r$r; done; done
         run p29-live-lfu-ahead-fuse-k3-c8 ;;
    p30) a=live-lfu-ahead-fuse-k3-min8
         for r in 1 2; do
           for q in live-lfu $a $a-side $a-stale2 $a-stale8 $a-stale32 $a-side-stale8; do run p30-$q-c16-r$r; done
         done
         for q in $a $a-side; do run p30-$q-c8; done
         run p30-$a-side-stale8-eval
         run p30-$a-side-a8l ;;
    # p32: p30's side arms again. plan() skipped the last MoE layer but join() still ran, so the
    # captured graph waited on a stream outside the capture and every side server died at start.
    # Also stale8's 379/401 split: two more draws beside K3 and live + LFU.
    p32) a=live-lfu-ahead-fuse-k3-min8
         for r in 1 2; do
           for q in $a-side live-lfu $a $a-stale8 $a-side-stale8; do run p32-$q-c16-r$r; done
         done
         run p32-$a-side-c8
         run p32-$a-side-stale8-eval
         run p32-$a-side-a8l ;;
    # p33: p31's chunk gain drawn again, 4k (less activation, more KV), ahead with 8k, and host
    # KV alone at 4k (it deadlocked at 8k), last so a stall cannot cost the rest.
    p33) for q in live-lfu-b8k-r2-a8l live-lfu-b4k-a8l live-lfu-ahead-b8k-a8l live-lfu-kvoff16-b8k-r2-a16l \
                  live-lfu-kvoff16-b4k-a16l base-kvoff8-b4k-a8l base-kvoff16-b4k-a16l; do run p33-$q; done ;;
    p37) for q in live-lfu-cg2-kvoff8-b8k-a8l live-lfu-cg2-kvoff16-b8k-a16l live-lfu-cg3-kvoff8-b8k-r2-a8l \
                  live-lfu-cg3-kvoff8-b4k-a8l live-lfu-cg3-kvoff16-b4k-a16l live-lfu-cg1-kvoff8-b8k-unjam-a8l; do
            run p37-$q; done ;;
    p38) for q in base-nolora-pref base-nolora-moefic-pref base-nolora-moeb12-pref base-nolora-moecut-pref; do
           run p38-$q; done ;;
    p38b) for q in base-nolora-moeb12-pref; do run p38-$q; done ;;
    p38c) for q in base-moeb12-lgate-eval base-moeb12-lgate-pref base-moeb12-lgate-kvoff8-b4k-unjam-a8l; do
            run p38-$q; done ;;
    p38d) for q in base-moeb12-lgate-r2-eval base-moeb12-lgate-kvoff16-b4k-unjam-a16l; do run p38-$q; done ;;
    p38e) for q in base-moeb12-lgate-exact-eval base-moeb12-lgate-exact-r2-eval base-moeb12-lgate-exact-pref; do
            run p38-$q; done ;;
    p39) for q in base-kvoff8-b4k-trace-a8l live-lfu-cg2-kvoff8-b8k-trace-a8l; do run p39-$q; done ;;
    p40) for q in base-kvoff8-b4k-a8l base-kvoff8-b4k-kvbf16-a8l base-kvoff8-b4k-triton-a8l \
                  base-kvoff8-b4k-fa2-kvbf16-a8l live-lfu-cg2-kvb270-kvoff8-b8k-a8l \
                  live-lfu-cg2-kvb270-kvoff8-b8k-cs8-a8l; do run p40-$q; done ;;
    # p41: p40's bf16 arms with bf16 KV (half the tokens per byte, so kvoff16 holds what kvoff8 did)
    p41) for q in base-kvoff8-b4k-kvbf16-a8l base-kvoff16-b4k-kvbf16-a8l \
                  base-kvoff16-b4k-fa2-kvbf16-a8l; do run p41-$q; done ;;
    p38f) for q in base-kvb105-kvoff8-b4k-unjam-a8l base-kvb105-kvoff16-b4k-unjam-a16l; do run p38-$q; done ;;   # Marlin at B12x's KV
    p36) for q in base-kvoff8-b6k-unjam-a8l base-kvoff16-b6k-unjam-a16l; do run p36-$q; done ;;
    p35) for q in live-lfu-cg3-b8k-a8l live-lfu-cg3-kvoff8-b8k-a8l live-lfu-cg3-kvoff16-b8k-a16l \
                  base-kvoff8-b4k-r2-a8l base-kvoff16-b4k-r2-a16l; do run p35-$q; done ;;
    p34) for q in base-kvoff8-b8k-ujdiag-a8l base-kvoff8-b8k-unjam-a8l base-kvoff16-b8k-unjam-a16l; do run p34-$q; done ;;
    p31) for q in base-kvoff8-a8l base-kvoff8-b8k-a8l live-lfu-a8l live-lfu-b8k-a8l live-lfu-b12k-a8l \
                  base-kvoff16-a16l base-kvoff16-b8k-a16l live-lfu-kvoff16-b8k-a16l; do run p31-$q; done
         for r in 1 2 3 4; do for q in live-lfu live-lfu-ahead; do run p31-$q-c1-r$r; done; done ;;
    p27) a=live-lfu-ahead-fuse
         for r in 1 2; do
           for q in live-lfu $a-k1 $a-k2 $a-k3; do run p27-$q-c16-r$r; done
           for c in 4 8; do for q in live-lfu $a-k2; do run p27-$q-c$c-r$r; done; done
         done
         for q in live-lfu $a-k2-min8; do run p27-$q-c1; done ;;
    p26) d=live-lfu-ahead-fuse-dry
         for q in live-lfu $d $d-noreq $d-noreq-nowait $d-nopred $d-nopred-noreq-nowait \
                  $d-nopred-noreq-nowait-nohelper $d-nopred-noreq-nowait-noah-nohelper; do
           run p26-$q-c1
         done
         for q in live-lfu live-lfu-ahead-fuse-k2 live-lfu-ahead-fuse-k3 live-lfu-ahead-fuse-k4 live-lfu-ahead-fuse; do
           run p26-$q-c16
         done ;;
    p25) for c in 1 4 16; do
           for q in live-lfu live-lfu-ahead-fuse-dry live-lfu-ahead-fuse-tick live-lfu-ahead-fuse; do
             run p25-$q-c$c
           done
         done ;;
    p18) for q in p18-ks-dma64-b1-ahead p18-ks-dma64-b1-ahead-eval p18-ks-dma64-b1-ahead-k10 \
                  p18-ks-dma64-b1-a8l p18-ks-dma64-b1-ahead-a8l; do run $q; done ;;
    *) run "$p" ;;
  esac
done
echo "== done $(date -u +%H:%M:%SZ)"

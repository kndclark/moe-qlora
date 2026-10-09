#!/usr/bin/env bash
# Row N, llama.cpp: Lightning NVFP4 converted to GGUF, served by llama-server, with the experts placed
# by llama.cpp's own flags instead of vLLM's. Three placements:
#   gpu         every weight on the card (-ngl 999)
#   ncm<N>      the experts of blocks 0..N-1 in RAM (--n-cpu-moe N counts all 52 blocks, 23 of them MoE:
#               ncm10 puts 4 MoE layers in RAM, ncm24 10, ncm52 all; l6's ncm10 kept 19 on the card)
#   cmoe        every expert in RAM (--cpu-moe); small batches run those experts on the CPU, larger
#               ones copy them over PCIe for each ubatch
#   ...-mc<M>   plus llama.cpp's GPU expert cache: M MiB of slots holding the most recently used
#               RAM experts (--moe-cache-mib, src/llama-moe-cache.cpp), uploaded on a miss
#   fit         no placement flags: llama.cpp's own fit (-fit on) picks; it aborts if -ngl is given
# The weights load without mmap (-lm none): with mmap, llama.cpp keeps RAM-placed weights in pageable
# memory (llama-model-loader.cpp, "avoid using a host buffer when using mmap"), which the GPU copies
# from more slowly than from pinned memory; -mmap arms test that default.
# KV is q8_0 (vLLM's arms use fp8), one 131,072-token slot per agent (-np, not unified). Mamba state
# cannot be rolled back, so llama-server saves checkpoints 4 and 4+ubatch tokens before a prompt's
# end and the next turn resumes from one; 4 per slot at about 48 MiB each. -cram 0 turns off the RAM
# prompt cache, which by default copies every idle slot (up to ~0.4 GiB at 127k) on each new task.
# -kvq4 arms store KV as q4_0 instead, about 3 GiB less at 16 slots, to give the MoE cache the room.
# No LoRA: p35's vLLM baselines served G6q. The server stops by SIGTERM so the MoE cache logs its
# hit rates, which it prints only on shutdown.
# Build: probes/llamacpp/build.sh. GGUF: convert_hf_to_gguf.py, see docs/laptop-memory-levers.md row N.
# Usage: probes/llamacpp/serve.sh ARM...
#   ARM = ll-{gpu|fit|ncm<N>|cmoe}[-mc<MiB>][-ub<N>][-t<N>][-omb<N>][-np<N>][-cram<MiB>][-kvq4][-mmap]-{smoke|pref|a8l|a16l}
#   -np<N> slots (default: one per agent, 16 for smoke and pref)
#   -t<N> threads (llama.cpp picks 4 on the 275HX's 8 P + 16 E cores)
#   -omb<N> GGML_OP_OFFLOAD_MIN_BATCH: ubatches of fewer tokens run the experts kept in RAM on the
#           CPU, larger ones copy them to the card (default 32, ggml-cuda.cu); the MoE cache's own
#           32 is a constant (llama-moe-cache.cpp max_batch) that this does not move
#   -cram<M> llama-server's RAM prompt cache at M MiB instead of off: with fewer slots than agents,
#           a request given the least recently used slot first saves that slot's state (KV, Mamba
#           state, checkpoints) to RAM and loads its own agent's, so KV for every agent need not fit
set -u
here=$(cd "$(dirname "$0")/../.." && pwd)
src=${LLAMA_SRC:-$HOME/llama.cpp-src/llama.cpp}
gguf=${GGUF:-$HOME/gguf/lightning-nvfp4.gguf}
J=$here/results/llamacpp; mkdir -p "$J"
B=http://127.0.0.1:8314
name=llamacpp-tm
M=lightning-nvfp4-gguf
[ -x "$src/build/bin/llama-server" ] || { echo "no $src/build/bin/llama-server: run probes/llamacpp/build.sh"; exit 9; }
[ -f "$gguf" ] || { echo "no $gguf"; exit 9; }
trap 'docker rm -f "$name" >/dev/null 2>&1' EXIT

phase() {  # $1 arm, $2 slots; the rest are llama-server flags
  local tag=$1 np=$2; shift 2
  echo "== $tag $(date -u +%H:%M:%SZ)"
  if sudo -n true 2>/dev/null; then   # pinned RAM wants free 2 MiB blocks (see kv_levers.sh phase)
    sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'
  fi
  local ram0=$(free -m | awk '/^Mem:/{print $3}')
  # the binary's RPATH is the build tree, so the source mounts at the path it was built under
  docker run -d --name "$name" --gpus all --ipc=host -p 127.0.0.1:8314:8080 "${envx[@]}" \
    -v "$src":/src:ro -v "$(dirname "$gguf")":/gguf:ro --entrypoint /src/build/bin/llama-server \
    --pull never vllm/vllm-openai:v0.29.0 \
    -m "/gguf/$(basename "$gguf")" --alias $M --host 0.0.0.0 --port 8080 --metrics \
    -np "$np" -c $((np * 131072)) -b 4096 -fa on -ctk q8_0 -ctv q8_0 -ctxcp 4 -cram 0 "$@" >/dev/null || return 1
  local t0=$(date +%s)
  until curl -sf $B/health >/dev/null; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
      echo "container exited before ready ($(( $(date +%s)-t0 ))s)"; docker logs "$name" > "$J/serve-$tag.log" 2>&1
      grep -iE "error|failed|abort" "$J/serve-$tag.log" | tail -4 | cut -c1-300
      docker rm -f "$name" >/dev/null 2>&1; return 2; fi
    [ $(( $(date +%s)-t0 )) -gt 1800 ] && { echo "not ready in 1800s"; docker logs "$name" > "$J/serve-$tag.log" 2>&1; docker rm -f "$name" >/dev/null 2>&1; return 3; }
    sleep 5
  done
  echo "ready in $(( $(date +%s)-t0 ))s; $(docker logs "$name" 2>&1 | grep -oE "(CUDA0|CUDA_Host|CPU|CPU_Mapped) (model|KV|RS|compute) buffer size = +[0-9.]+ MiB|n_threads = [0-9]+ / [0-9]+" | sed 's/  */ /g' | tr '\n' ';')"
  docker logs "$name" 2>&1 | grep -E "llama_moe_cache|MoE cache|pipeline parallelism|fit" | grep -v "^print_info" | head -6 | sed 's/^/  /' | cut -c1-200
  echo "  GPU used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"   # buffer sizes log only at -lv 4
  echo "  host RAM used: +$(( $(free -m | awk '/^Mem:/{print $3}') - ram0 )) MiB, $(free -m | awk '/^Mem:/{print $7}') MiB available"
  local p
  for p in "2+2=" "The capital of France is"; do
    echo "  smoke $p $(curl -s $B/v1/completions -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"prompt\":\"$p\",\"max_tokens\":8,\"temperature\":0}" | python3 -c 'import json,sys; print(repr(json.load(sys.stdin)["choices"][0]["text"]))' 2>&1)"
  done
  if [ -n "${WORK:-}" ]; then
    local t=$(date +%s)
    # shellcheck disable=SC2086  # WORK and BARGS are word lists
    # the client gets BENCH_S (40 minutes); a stall ends the bench, never the container (its log is kept)
    timeout "${BENCH_S:-2400}" python3 "$here/probes/offload_bench.py" --base $B --model $M --out "$J/bench-$tag.json" $WORK ${BARGS:-} \
      > "$J/bench-$tag.log" 2>&1
    echo "  bench: exit $?, $(( $(date +%s)-t ))s"; grep -E "^  (decode|prefill|agent)" "$J/bench-$tag.log"
  fi
  docker stop -t 120 "$name" >/dev/null 2>&1   # SIGTERM: the MoE cache logs its stats on the way out
  docker logs "$name" > "$J/serve-$tag.log" 2>&1
  grep -E "llama_moe_cache: (ubatch|large)" "$J/serve-$tag.log" | sed 's/^/  /'
  docker rm -f "$name" >/dev/null 2>&1
}

run() {
  local a=$1 o=(-ngl 999 -fit off) np=16 x envx=()
  case $a in
    ll-gpu-*) ;;
    ll-fit-*) o=() ;;
    ll-cmoe-*) o+=(-cmoe) ;;
    ll-ncm[0-9]*) x=${a#ll-ncm}; o+=(-ncmoe "${x%%-*}") ;;
    *) echo "bad arm $a"; return 9 ;;
  esac
  # the cache logs its size and hit rates at info level, which the default (-lv 3) hides (l3)
  case $a in *-mc[0-9]*) x=${a#*-mc}; o+=(--moe-cache-mib "${x%%-*}" -lv 4) ;; esac
  case $a in *-ub[0-9]*) x=${a#*-ub}; o+=(-ub "${x%%-*}") ;; esac
  case $a in *-t[0-9]*) x=${a#*-t}; o+=(-t "${x%%-*}" -tb "${x%%-*}") ;; esac
  case $a in *-omb[0-9]*) x=${a#*-omb}; envx+=(-e "GGML_OP_OFFLOAD_MIN_BATCH=${x%%-*}") ;; esac
  # after phase's -cram 0; -lv 4 logs each cache update (save, load, state) and its time
  case $a in *-cram[0-9]*) x=${a#*-cram}; o+=(-cram "${x%%-*}" -lv 4) ;; esac
  case $a in *-mmap-*) o+=(-lm mmap) ;; *) o+=(-lm none) ;; esac
  case $a in *-kvq4-*) o+=(-ctk q4_0 -ctv q4_0) ;; esac   # after phase's q8_0: the last flag wins
  case $a in *-a8l) np=8 ;; esac
  case $a in *-np[0-9]*) x=${a#*-np}; np=${x%%-*} ;; esac   # fewer slots than agents queue the rest
  case $a in
    *-smoke) phase "$a" $np "${o[@]}" ;;
    *-pref) WORK="decode prefill" BARGS="--conc 1,$np --lens 8192,32768,98304" phase "$a" $np "${o[@]}" ;;
    *-a8l) WORK=agent BARGS="--agents 8 --start 8192 --chunk 4096 --gen 256 --target 126976" phase "$a" $np "${o[@]}" ;;
    *-a16l) WORK=agent BARGS="--agents 16 --start 8192 --chunk 4096 --gen 256 --target 126976" phase "$a" $np "${o[@]}" ;;
    *) echo "bad arm $a"; return 9 ;;
  esac
}
for a in "$@"; do run "$a"; done

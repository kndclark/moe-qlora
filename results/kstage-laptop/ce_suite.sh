#!/bin/bash
# Laptop, 2026-10-06. Row O: ce_probe, one config at a time. The first run (ce.out's head) hung in
# g.replay with ~20 steps queued, after K=1's first replay had worked; so the measurements cap the
# steps in flight (--inflight 2 and 1, about what vLLM queues), and the no-cap runs come last, as
# diagnostics: does the watchdog find the helper inside cuMemcpyHtoDAsync, and do the levers help.
# Each run is bounded three ways: the helper's CE_WATCH watchdog (20 s inside one CUDA call), the
# probe's faulthandler backstop (200 s), and timeout -k around docker run --init (Python as PID 1
# ignores SIGTERM; the first laptop run outlived timeout 300 that way). Then row P's CPU timing,
# which needs an idle machine. Then p15, but only if no probe container is left on the GPU (its
# memory would shrink p15's KV).
cd "$(dirname "$0")/../../probes/kstage" || exit 1
res=../../results/kstage-laptop
ce() {  # name, "env ...", probe args...
  local name=$1 env=$2; shift 2
  local envs=(); for e in $env; do envs+=(-e "$e"); done
  echo "=== $name ($env $*) $(date +%T)" >> $res/ce.out
  timeout -k 10 300 docker run --rm --init --name "ce-$name" --gpus all --pull never "${envs[@]}" \
    --entrypoint python3 -v "$PWD":/p:ro vllm/vllm-openai:v0.29.0 /p/ce_probe.py --timeout 20 "$@" \
    2>&1 | grep -v "^WARNING\|^INFO\|^$" >> $res/ce.out
  echo "=== $name exit ${PIPESTATUS[0]} $(date +%T)" >> $res/ce.out
  sleep 5
  if [ -n "$(docker ps -q -f name=ce-$name)" ]; then echo "=== ce-$name still running: stopping the suite" >> $res/ce.out; exit 1; fi
}
ce inflight2 "" --inflight 2 --k 0,1,2,3 --steps 50
ce inflight1 "" --inflight 1 --k 0,1,2,3 --steps 50
ce nocap "" --inflight 0 --k 1 --steps 20
ce nocap-conn32 "CUDA_DEVICE_MAX_CONNECTIONS=32" --inflight 0 --k 1 --steps 20
ce nocap-prio "CE_PRIO=1" --inflight 0 --k 1 --steps 20
echo "=== row P $(date +%T)" >> $res/cpu_expert.out
timeout -k 10 1200 docker run --rm --init --pull never --entrypoint python3 -v /srv/model-cache:/hf:ro -v "$PWD":/p:ro \
  vllm/vllm-openai:v0.29.0 /p/cpu_expert_probe.py --time --threads 1,4,8,16,24 2>&1 \
  | grep -v "^WARNING\|^INFO\|^$" >> $res/cpu_expert.out
echo "=== row P exit ${PIPESTATUS[0]} $(date +%T)" >> $res/cpu_expert.out
cd ../.. && bash probes/kv_levers.sh p15 > results/kv-levers/p15.out 2>&1

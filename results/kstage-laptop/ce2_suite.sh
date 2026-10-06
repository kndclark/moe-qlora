#!/bin/bash
# Laptop, 2026-10-06. Row O, second round: what a fill-ahead costs when most layers fetch nothing.
# A decode fill-ahead fetches only predicted misses, so most layers' plans are empty; --devdone
# lets the graph mark those done itself (no host round trip), --sparse gives rows to a fraction of
# layers. The first run repeats ce.out's inflight1 baseline in the same session. Same three bounds
# as ce_suite.sh (CE_WATCH, faulthandler backstop, timeout -k around docker run --init).
cd "$(dirname "$0")/../../probes/kstage" || exit 1
res=../../results/kstage-laptop
ce() {  # name, probe args...
  local name=$1; shift
  echo "=== $name ($*) $(date +%T)" >> $res/ce2.out
  timeout -k 10 300 docker run --rm --init --name "ce-$name" --gpus all --pull never \
    --entrypoint python3 -v "$PWD":/p:ro vllm/vllm-openai:v0.29.0 /p/ce_probe.py --timeout 20 "$@" \
    2>&1 | grep -v "^WARNING\|^INFO\|^$" >> $res/ce2.out
  echo "=== $name exit ${PIPESTATUS[0]} $(date +%T)" >> $res/ce2.out
  sleep 5
  if [ -n "$(docker ps -q -f name=ce-$name)" ]; then echo "=== ce-$name still running: stopping the suite" >> $res/ce2.out; exit 1; fi
}
ce base1 --inflight 1 --k 0,1,2 --steps 50
ce dd1 --inflight 1 --devdone --k 0,1,2 --steps 50
ce dd1-s50 --inflight 1 --devdone --sparse 0.5 --k 1,2,3 --steps 50
ce dd1-s25 --inflight 1 --devdone --sparse 0.25 --k 1,2,3 --steps 50
ce dd2-s25 --inflight 2 --devdone --sparse 0.25 --k 1,2,3 --steps 50

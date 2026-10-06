#!/bin/bash
# Laptop, 2026-10-06. Row O: the KSTAGE_AHEAD deadlock outside vLLM (probes/kstage/ce_lock.c).
# Three helper placements x three blocking main-thread calls, then whether each placement's
# copy-engine traffic overlaps a kernel on every SM. Each run ends itself (alarm 10 s).
cd "$(dirname "$0")/../../probes/kstage" || exit 1
out=../../results/kstage-laptop/ce_lock.out
echo "=== $(date -u +%FT%TZ)" >> $out
timeout -k 10 600 docker run --rm --init --name ce-lock --gpus all --pull never --entrypoint bash \
  -v "$PWD":/p:ro vllm/vllm-openai:v0.29.0 -c '
  gcc -O2 -Wall /p/ce_lock.c -o /tmp/ce_lock -I/usr/local/cuda/include -L/usr/local/cuda/lib64/stubs -lcuda -lpthread || exit 1
  for m in same ctx proc; do for op in d2h sync free overlap; do timeout -k 5 60 /tmp/ce_lock $m $op 2>&1; echo "--- $m $op exit $?"; done; done' \
  >> $out 2>&1
echo "=== exit $?" >> $out

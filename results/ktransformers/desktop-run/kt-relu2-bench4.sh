#!/bin/bash
# Runs ON the desktop: relu2 CPU-expert throughput, AVX2 GPTQ int4 x 5/10/20 threads (no AVX-VNNI on this CPU)
set -uo pipefail
OUT=$HOME/kt-relu2-results
mkdir -p "$OUT"
docker cp "$HOME/kt-relu2-src/kt-kernel/bench/bench_relu2_moe.py" kt-relu2:/src/kt-kernel/bench/bench_relu2_moe.py || exit 1
for t in 5 10 20; do
  f="$OUT/desktop-gptq4-t$t"
  docker exec kt-relu2 python3 /src/kt-kernel/bench/bench_relu2_moe.py --backend gptq4 --threads $t > "$f.jsonl" 2> "$f.err"
  echo "gptq4 t$t rc=$?"
done
echo bench_done

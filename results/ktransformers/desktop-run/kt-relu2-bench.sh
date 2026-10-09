#!/bin/bash
# Runs ON the desktop: relu2 CPU-expert throughput, both backends x 5/10/20 threads x the full token-count list
set -uo pipefail
OUT=$HOME/kt-relu2-results
mkdir -p "$OUT"
docker cp "$HOME/kt-relu2-src/kt-kernel/bench/bench_relu2_moe.py" kt-relu2:/src/kt-kernel/bench/bench_relu2_moe.py || exit 1
for b in bf16 fp8; do
  for t in 5 10 20; do
    f="$OUT/desktop-$b-t$t"
    docker exec kt-relu2 python3 /src/kt-kernel/bench/bench_relu2_moe.py --backend $b --threads $t > "$f.jsonl" 2> "$f.err"
    echo "$b t$t rc=$?"
  done
done
echo bench_done

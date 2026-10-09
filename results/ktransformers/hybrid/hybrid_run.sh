#!/bin/bash
# Contention test, runs on one node: CPU relu2 experts (kt-relu2 container) vs a concurrent
# H2D copy load (its own GPU container). Usage: hybrid_run.sh <dir> <node> <backend> <threads...>
set -uo pipefail
D=$1; NODE=$2; B=$3; shift 3
IMG=vllm/vllm-openai:v0.29.0
cd "$D" || exit 1
copy_start() {  # $1 = output name
  rm -f "$D/$1"
  docker run -d --rm --name h2d-load --pull never --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
    --entrypoint python3 -v "$D:/w" $IMG -u /w/h2d_load.py --out "/w/$1" > /dev/null || return 1
  for i in $(seq 1 120); do grep -q ready "$D/$1" 2>/dev/null && return 0; sleep 1; done
  return 1
}
copy_stop() { docker rm -f h2d-load > /dev/null 2>&1; }
kt() {  # $1 threads, $2 output name
  docker exec kt-relu2 python3 -u /src/kt-kernel/bench/bench_relu2_moe.py --backend $B --threads $1 \
    --qlens 1,2,4,8,16 --read-probe-gib 0 2> "$D/$2.err" \
    | while IFS= read -r l; do printf '%s\t%s\n' "$(date +%s.%N)" "$l"; done > "$D/$2.tsv"
}
copy_stop
copy_start "$NODE-copy-solo-a.jsonl" && sleep 15; copy_stop; echo "copy solo a done"
for t in "$@"; do
  kt $t "$NODE-$B-t$t-solo"; echo "kt t$t solo done"
  copy_start "$NODE-copy-during-$B-t$t.jsonl" || { echo "copy load failed to start"; copy_stop; continue; }
  sleep 3
  kt $t "$NODE-$B-t$t-hybrid"; echo "kt t$t hybrid done"
  sleep 2; copy_stop
done
copy_start "$NODE-copy-solo-b.jsonl" && sleep 15; copy_stop; echo "copy solo b done"
echo hybrid_done

#!/bin/bash
# K6's copy kernel, after the k10 suite. The default copy launches a program per 512 words of
# every possible copy (min(slots, routings, E) rows a step, missed or not); KSTAGE_COPY=N runs N
# programs that walk the copy list. cache_bench.py times both without a server (the fixed cost of
# a layer-step with every expert in a slot, one copy's cost over a trace, copies beside compute);
# the fastest N at 64 slots and 16 streams then runs the decode arms. Same bytes copied, so the
# greedy outputs should match the default kernel's twins (k10's, and one rerun here for noise).
cd ~/kstage-tmp || exit 1
until grep -q "k10 done" k10.out 2>/dev/null || ! pgrep -f "[k]10_suite" >/dev/null; do sleep 30; done
[ "$(md5sum < vllm_kstage.py)" = "f9ed6abbb2f9c28d39460895fb454ab7  -" ] || { echo "plugin is not k10's; k11 stopped"; exit 1; }
cp vllm_kstage.py vllm_kstage.k10.py && cp k11/vllm_kstage.py vllm_kstage.py
echo "== cache_bench $(date -u +%H:%M:%SZ)"
docker run --rm --gpus all --ipc=host -v "$PWD":/k:ro -w /k/k11 --entrypoint python3 --pull never \
  vllm/vllm-openai:v0.29.0 /k/k11/cache_bench.py /k/k6/bench-p7-experts.json /k/p7-all.json \
  --layers 4 --conc 1,4,16 --slots all,64,16 --few 32,82,164,328 2>&1 | grep -v Warning | tee k11.bench.txt
echo "cache_bench rc=${PIPESTATUS[0]} $(date -u +%H:%M:%SZ)"
N=$(awk '/^slots 64 C=16 few/ { for (i = 1; i <= NF; i++) if ($i == "trace") t = $(i + 1) + 0
          n = $4; sub("few", "", n); sub(":", "", n); if (b == "" || t < b) { b = t; bn = n } }
         END { print bn }' k11.bench.txt)
[ -n "$N" ] || N=82   # cache_bench failed: one program per SM
echo "KSTAGE_COPY=$N"
P="KSTAGE=cache KSTAGE_COLD_GB=4 KSTAGE_PROFILE=/k/p7-all.json KSTAGE_STATS=30"
./kstage_check.sh cache4-s64-c$N "$P KSTAGE_SLOTS=64 KSTAGE_COPY=$N"
./kstage_check.sh cache4-s64-g2 "$P KSTAGE_SLOTS=64"   # the default kernel again: run-to-run noise
./kstage_check.sh cache4-s16-c$N "$P KSTAGE_SLOTS=16 KSTAGE_COPY=$N"
avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); echo "MemAvailable ${avail} GiB"
if [ "$avail" -ge 22 ]; then   # slots all: ~16.5 GB of host homes, pinned
  ./kstage_check.sh cache4-all-c$N "$P KSTAGE_COPY=$N"
else echo "skip cache4-all: ~16.5 GB pinned homes need >= 22 GiB available"; fi
REF=base-mx python3 k9/lpcmp3.py cache4-s64-c$N cache4-s64-g2 cache4-s16-c$N cache4-all-c$N
REF=cache4-s64-mx python3 k9/lpcmp3.py cache4-s64-c$N cache4-s64-g2
REF=cache4-s16-mx python3 k9/lpcmp3.py cache4-s16-c$N
REF=cache4-all-mx python3 k9/lpcmp3.py cache4-all-c$N
echo "k11 done $(date -u +%H:%M:%SZ)"

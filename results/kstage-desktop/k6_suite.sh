#!/bin/bash
# K6 (KSTAGE=cache) on the desktop, after det_suite: install the new plugin, gate on the GPU
# cache test (C=16,4), serve arms, then the slow C=1 full-trace test.
cd ~/kstage-tmp || exit 1
until grep -q "det done" det.out 2>/dev/null; do sleep 30; done
cp vllm_kstage.py vllm_kstage.k5.py && cp k6/vllm_kstage.py vllm_kstage.py
ct() { docker run --rm --gpus all --ipc=host -v "$PWD":/k:ro -w /k/k6 --entrypoint python3 --pull never \
         vllm/vllm-openai:v0.29.0 /k/k6/cache_test.py /k/k6/bench-p7-experts.json /k/p7-all.json "$@"; }
echo "== cache_test C=16,4 $(date -u +%H:%M:%SZ)"
ct --conc 16,4 --slots all,16,64; rc=$?; echo "cache_test rc=$rc $(date -u +%H:%M:%SZ)"
[ $rc = 0 ] || { echo "k6 stopped"; exit 1; }
P="KSTAGE=cache KSTAGE_COLD_GB=4 KSTAGE_PROFILE=/k/p7-all.json KSTAGE_STATS=30"
./kstage_check.sh cache4-s16 "$P KSTAGE_SLOTS=16"
./kstage_check.sh cache4-s64 "$P KSTAGE_SLOTS=64"
avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); echo "MemAvailable ${avail} GiB"
if [ "$avail" -ge 22 ]; then   # slots all: ~16.5 GB of host homes, pinned
  ./kstage_check.sh cache4-all "$P"
  ./kstage_check.sh cache4-all-freeze "$P KSTAGE_FREEZE_M=256"
else echo "skip cache4-all: ~16.5 GB pinned homes need >= 22 GiB available"; fi
echo "== cache_test C=1 $(date -u +%H:%M:%SZ)"
ct --conc 1 --slots all,16,64; echo "cache_test C=1 rc=$? $(date -u +%H:%M:%SZ)"
echo "k6 done"

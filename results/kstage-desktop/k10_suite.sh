#!/bin/bash
# K6 LFU eviction (KSTAGE_EVICT=lfu) and a mixed-prompt decode, after the k9 suite: install the
# new plugin and client, gate on the GPU cache test (LFU against its reference and the simulator,
# LRU unchanged), then LFU and LRU arms with the mixed decode, then the slow C=1 LFU trace.
cd ~/kstage-tmp || exit 1
until grep -q "k9 done" k9.out 2>/dev/null || ! pgrep -f "[k]9_suite" >/dev/null; do sleep 30; done
[ "$(md5sum < vllm_kstage.py)" = "4381cfe54f4c600632a712e1dad2afb9  -" ] || { echo "plugin changed since k8; k10 stopped"; exit 1; }
cp vllm_kstage.py vllm_kstage.k8.py && cp kstage_client.py kstage_client.k8.py
cp k10/vllm_kstage.py vllm_kstage.py && cp k10/kstage_client.py kstage_client.py
ct() { docker run --rm --gpus all --ipc=host -v "$PWD":/k:ro -w /k/k10 --entrypoint python3 --pull never \
         vllm/vllm-openai:v0.29.0 /k/k10/cache_test.py /k/k6/bench-p7-experts.json /k/p7-all.json "$@"; }
echo "== lora_fix_test $(date -u +%H:%M:%SZ)"
docker run --rm -v "$PWD":/k:ro -w /k/k10 --entrypoint python3 --pull never vllm/vllm-openai:v0.29.0 \
  /k/k10/lora_fix_test.py 2>&1 | tail -1
echo "== cache_test C=16,4 $(date -u +%H:%M:%SZ)"
ct --evict lru,lfu --conc 16,4 --slots all,16,64; rc=$?; echo "cache_test rc=$rc $(date -u +%H:%M:%SZ)"
[ $rc = 0 ] || { echo "k10 stopped"; exit 1; }
P="KSTAGE=cache KSTAGE_COLD_GB=4 KSTAGE_PROFILE=/k/p7-all.json KSTAGE_STATS=30"
./kstage_check.sh base-mx ""
./kstage_check.sh cache4-s64-lfu "$P KSTAGE_SLOTS=64 KSTAGE_EVICT=lfu"
./kstage_check.sh cache4-s64-mx "$P KSTAGE_SLOTS=64"
./kstage_check.sh cache4-s16-lfu "$P KSTAGE_SLOTS=16 KSTAGE_EVICT=lfu"
./kstage_check.sh cache4-s16-mx "$P KSTAGE_SLOTS=16"
avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); echo "MemAvailable ${avail} GiB"
if [ "$avail" -ge 22 ]; then   # slots all: ~16.5 GB of host homes, pinned
  ./kstage_check.sh cache4-all-lfu "$P KSTAGE_EVICT=lfu"
  ./kstage_check.sh cache4-all-mx "$P"
else echo "skip cache4-all: ~16.5 GB pinned homes need >= 22 GiB available"; fi
./kstage_check.sh off4-mx "" --cpu-offload-params experts --cpu-offload-gb 4
python3 k9/lpcmp3.py base-mx cache4-s64-lfu cache4-s64-mx cache4-s16-lfu cache4-s16-mx cache4-all-lfu cache4-all-mx off4-mx
echo "== cache_test C=1 $(date -u +%H:%M:%SZ)"
ct --evict lfu --conc 1 --slots all,16,64; echo "cache_test C=1 rc=$? $(date -u +%H:%M:%SZ)"
echo "k10 done $(date -u +%H:%M:%SZ)"

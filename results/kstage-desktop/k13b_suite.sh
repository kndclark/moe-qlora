#!/bin/bash
# k13 again after the fix (_install_cache copied 16 rows into the last chunk of R; staging makes
# the tensor T > R rows), with KSTAGE_DMA_TIME=1: per big step, copy-engine busy time, GiB, GB/s
# and the MoE layers' span. k13's dma64-s64 prefill was 3.58 s vs cache4-s64-c32 3.23 s: the
# timing says whether the staging copy runs at the link's 12.2 GB/s and overlaps compute.
# Then row O's mechanism probe (ce_probe.py) on the idle GPU.
cd ~/kstage-tmp || exit 1
[ "$(md5sum < vllm_kstage.py)" = "f938d6fd5435584f2c294aad1b4bf5c0  -" ] || { echo "plugin is not k13's; k13b stopped"; exit 1; }
cp k13b/vllm_kstage.py vllm_kstage.py
P="KSTAGE=cache KSTAGE_PROFILE=/k/p7-all.json KSTAGE_STATS=30 KSTAGE_COPY=32 KSTAGE_COLD_GB=4 KSTAGE_DMA_M=64 KSTAGE_DMA_TIME=1"
./kstage_check.sh dma64t-s64 "$P KSTAGE_SLOTS=64"
avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); echo "MemAvailable ${avail} GiB"
if [ "$avail" -ge 22 ]; then
  ./kstage_check.sh dma64t-all "$P"
  ./kstage_check.sh dma64t-all-b1 "$P KSTAGE_DMA_BUF=1"
else echo "skip slots all: ~16.5 GB pinned homes need >= 22 GiB available"; fi
for t in base-mx cache4-s64-c32 cache4-all-c32 dma64-s64 dma64t-s64 dma64t-all dma64t-all-b1; do
  [ -f out/$t.json ] && python3 -c "import json; r = json.load(open('out/$t.json'))
print(f\"{'$t':16s} decode {r['decode']} mixed {r['decode_mixed']} prefill {r['prefill_s']} s\")"
  grep -h "DMA from\|kstage: dma\|Traceback\|Error" out/serve-$t.log 2>/dev/null | cut -c1-260 | tail -8
done
REF=base-mx python3 k9/lpcmp3.py dma64t-s64 dma64t-all dma64t-all-b1
REF=cache4-s64-mx python3 k9/lpcmp3.py dma64t-s64
REF=cache4-all-mx python3 k9/lpcmp3.py dma64t-all dma64t-all-b1
echo "== ce_probe $(date -u +%H:%M:%SZ)"
docker run --rm --gpus all --pull never --entrypoint python3 -v ~/kstage-tmp/k14:/p vllm/vllm-openai:v0.29.0 /p/ce_probe.py 2>&1 | tail -20
echo "k13b done $(date -u +%H:%M:%SZ)"

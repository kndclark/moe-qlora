#!/bin/bash
# P1: K6 with KSTAGE_DMA_M. Batches of 64+ tokens outside CUDA graphs (prefill chunks) copy every
# non-resident expert into staging rows on the copy engine, KSTAGE_DMA_BUF layers ahead, instead
# of Marlin reading ~33 experts a layer through UVA. Correctness here, speed on the laptop: the
# desktop copies at 12.2 GB/s (~330 ms for 4 GiB a step), the laptop at 51.8.
#   dma64-s64, dma64-all: default two buffers; dma64-all-b1: one (copy and compute alternate).
# Staged rows are exact copies of the homes, so greedy outputs should match cache4-*-mx up to the
# noise twins show (k10, k11). Prefill reference: base-mx 1.14 s, cache4-all-c32 2.7 s.
cd ~/kstage-tmp || exit 1
until grep -q "k12 done" k12.out 2>/dev/null || ! pgrep -f "[k]12_suite" >/dev/null; do sleep 30; done
[ "$(md5sum < vllm_kstage.py)" = "b0fc83e8e0482d5c8f21e3abbb9f9761  -" ] || { echo "plugin is not k12's; k13 stopped"; exit 1; }
cp vllm_kstage.py vllm_kstage.k12.py && cp k13/vllm_kstage.py vllm_kstage.py
P="KSTAGE=cache KSTAGE_PROFILE=/k/p7-all.json KSTAGE_STATS=30 KSTAGE_COPY=32 KSTAGE_COLD_GB=4 KSTAGE_DMA_M=64"
./kstage_check.sh dma64-s64 "$P KSTAGE_SLOTS=64"
avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); echo "MemAvailable ${avail} GiB"
if [ "$avail" -ge 22 ]; then   # slots all: a pinned host home for every expert, ~16.5 GB
  ./kstage_check.sh dma64-all "$P"
  ./kstage_check.sh dma64-all-b1 "$P KSTAGE_DMA_BUF=1"
else echo "skip slots all: ~16.5 GB pinned homes need >= 22 GiB available"; fi
for t in base-mx cache4-s64-c32 cache4-all-c32 dma64-s64 dma64-all dma64-all-b1; do
  [ -f out/$t.json ] && python3 -c "import json; r = json.load(open('out/$t.json'))
print(f\"{'$t':16s} decode {r['decode']} mixed {r['decode_mixed']} prefill {r['prefill_s']} s\")"
  grep -h "DMA from\|kstage: dma:\|Traceback\|Error" out/serve-$t.log 2>/dev/null | grep -v "^.*kstage: model" | tail -4 | cut -c1-240
done
REF=base-mx python3 k9/lpcmp3.py dma64-s64 dma64-all dma64-all-b1
REF=cache4-s64-mx python3 k9/lpcmp3.py dma64-s64
REF=cache4-all-mx python3 k9/lpcmp3.py dma64-all dma64-all-b1
echo "k13 done $(date -u +%H:%M:%SZ)"

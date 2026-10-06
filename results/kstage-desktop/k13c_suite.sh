#!/bin/bash
# P1 for real. k13/k13b's staging never used the copy engine: _install_cache asked _is_big(t0, E)
# after _set_rows had made t0 T rows, so every expert tensor went to the device index_select
# list (SMs reading host pages through UVA) and the copy list was empty (k13b: 0.00 GiB a step).
# Same arms with the classification fixed; k13b's dma64t-* are the SM-gather comparison.
cd ~/kstage-tmp || exit 1
until grep -q "k13b done" k13b.out; do sleep 20; done
[ "$(md5sum < vllm_kstage.py)" = "346a30a238eb0f3166ff8f5d2f627c18  -" ] || { echo "plugin is not k13b's; k13c stopped"; exit 1; }
cp k13c/vllm_kstage.py vllm_kstage.py
P="KSTAGE=cache KSTAGE_PROFILE=/k/p7-all.json KSTAGE_STATS=30 KSTAGE_COPY=32 KSTAGE_COLD_GB=4 KSTAGE_DMA_M=64 KSTAGE_DMA_TIME=1"
./kstage_check.sh dma64c-s64 "$P KSTAGE_SLOTS=64"
avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); echo "MemAvailable ${avail} GiB"
if [ "$avail" -ge 22 ]; then
  ./kstage_check.sh dma64c-all "$P"
  ./kstage_check.sh dma64c-all-b1 "$P KSTAGE_DMA_BUF=1"
else echo "skip slots all: ~16.5 GB pinned homes need >= 22 GiB available"; fi
for t in base-mx cache4-s64-c32 cache4-all-c32 dma64t-s64 dma64c-s64 dma64t-all dma64c-all dma64t-all-b1 dma64c-all-b1; do
  [ -f out/$t.json ] && python3 -c "import json; r = json.load(open('out/$t.json'))
print(f\"{'$t':16s} decode {r['decode']} mixed {r['decode_mixed']} prefill {r['prefill_s']} s\")"
  grep -h "DMA from\|Traceback\|Error" out/serve-$t.log 2>/dev/null | cut -c1-200 | tail -3
  grep -h "kstage: dma step" out/serve-$t.log 2>/dev/null | cut -c1-200 | tail -6
done
REF=base-mx python3 k9/lpcmp3.py dma64c-s64 dma64c-all dma64c-all-b1
REF=cache4-s64-mx python3 k9/lpcmp3.py dma64c-s64
REF=cache4-all-mx python3 k9/lpcmp3.py dma64c-all dma64c-all-b1
echo "k13c done $(date -u +%H:%M:%SZ)"

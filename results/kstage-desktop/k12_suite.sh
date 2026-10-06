#!/bin/bash
# Where K6's per-layer-step cost goes. k11: base 204.5 tok/s at c=1, slots all 135.2, slots 64
# 127.0. Three arms take one cost away each:
#   cold0-*: KSTAGE_COLD_GB=0 puts no expert in host memory, so every unpinned expert starts in
#     its own slot and nothing can miss. What is left is the fixed machinery: the cache kernel,
#     the id remap, the empty copy launch, and Marlin seeing E + S experts instead of E.
#   frz1: KSTAGE_FREEZE_M=1 freezes every step (S = 0: no copies, no copy launch). The slots keep
#     the profile's starting fill and every other cold expert is read over UVA, as K5 does.
#   cold0-frz1: both, which leaves the cache kernel's remap and the E + S expert count alone.
# The cold0 arms copy nothing and read nothing from host, so their greedy outputs should match
# base-mx up to the noise twins show (k10, k11).
cd ~/kstage-tmp || exit 1
until ! pgrep -f "[k]11_suite" >/dev/null; do sleep 30; done
[ "$(md5sum < vllm_kstage.py)" = "b0fc83e8e0482d5c8f21e3abbb9f9761  -" ] || { echo "plugin is not k11's; k12 stopped"; exit 1; }
P="KSTAGE=cache KSTAGE_PROFILE=/k/p7-all.json KSTAGE_STATS=30 KSTAGE_COPY=32"
./kstage_check.sh cold0-s16 "$P KSTAGE_COLD_GB=0 KSTAGE_SLOTS=16"
./kstage_check.sh cold0-frz1 "$P KSTAGE_COLD_GB=0 KSTAGE_SLOTS=16 KSTAGE_FREEZE_M=1"
./kstage_check.sh cache4-s64-frz1 "$P KSTAGE_COLD_GB=4 KSTAGE_SLOTS=64 KSTAGE_FREEZE_M=1"
avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); echo "MemAvailable ${avail} GiB"
if [ "$avail" -ge 22 ]; then   # slots all: a pinned host home for every expert, ~16.5 GB
  ./kstage_check.sh cold0-all "$P KSTAGE_COLD_GB=0"
  ./kstage_check.sh cache4-all-frz1 "$P KSTAGE_COLD_GB=4 KSTAGE_FREEZE_M=1"
else echo "skip slots all: ~16.5 GB pinned homes need >= 22 GiB available"; fi
for t in base-mx cache4-s16-c32 cache4-s64-c32 cache4-all-c32 cold0-s16 cold0-frz1 cache4-s64-frz1 cold0-all cache4-all-frz1; do
  [ -f out/$t.json ] && python3 -c "import json; r = json.load(open('out/$t.json'))
print(f\"{'$t':16s} decode {r['decode']} mixed {r['decode_mixed']} prefill {r['prefill_s']} s\")"
  grep -h "kstage: cache:.*misses" out/serve-$t.log 2>/dev/null | tail -1 | cut -c1-200
done
REF=base-mx python3 k9/lpcmp3.py cold0-s16 cold0-frz1 cold0-all cache4-s64-frz1 cache4-all-frz1
echo "k12 done $(date -u +%H:%M:%SZ)"

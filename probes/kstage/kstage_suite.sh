#!/bin/bash
# Correctness + speed of the kstage modes against plain vLLM, one config after another.
cd "$(dirname "$0")"
UVA="--cpu-offload-params experts --cpu-offload-gb 4"
./kstage_check.sh base ""
./kstage_check.sh off4 "" $UVA
./kstage_check.sh off4-gather "KSTAGE=gather" $UVA
./kstage_check.sh off4-hotcold "KSTAGE=hotcold" $UVA
./kstage_check.sh hotcold4 "KSTAGE=hotcold KSTAGE_COLD_GB=4"
echo "suite done $(date -u +%H:%M:%SZ)"

#!/bin/bash
# vLLM's prefetch offloader (copies whole MoE layers ahead on the copy engine)
# against the same greedy ids and speeds as the UVA and kstage configs.
# Group size 4, last one of each group: decoder layers 3,7,...,51, of which
# MoE layers 3,15,27,31,43,47,51 hold experts (7 x 0.72 GB, ~4.3 GB net).
cd "$(dirname "$0")"
PF="--offload-backend prefetch --offload-num-in-group 1 --offload-params experts"
./kstage_check.sh pf4 "" $PF --offload-group-size 4
./kstage_check.sh pf4-s2 "" $PF --offload-group-size 4 --offload-prefetch-step 2
./kstage_check.sh off4-gdma "KSTAGE=gather KSTAGE_DMA_M=256" --cpu-offload-params experts --cpu-offload-gb 4   # whole-layer DMA for prefill
echo "pf done"

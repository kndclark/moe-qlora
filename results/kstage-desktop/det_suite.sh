#!/bin/bash
# Is prefetch's token-0 flip a race? Rerun pf4/pf4-s2 and compare to themselves; logprobs on the
# references show whether token 0 is a near-tie. Then hotcold with a real routing profile.
cd "$(dirname "$0")"
PF="--offload-backend prefetch --offload-num-in-group 1 --offload-params experts --offload-group-size 4"
UVA="--cpu-offload-params experts --cpu-offload-gb 4"
./kstage_check.sh base-lp ""
./kstage_check.sh off4-lp "" $UVA
./kstage_check.sh pf4-s2-r2 "" $PF --offload-prefetch-step 2
./kstage_check.sh pf4-s2-r3 "" $PF --offload-prefetch-step 2
./kstage_check.sh pf4-r2 "" $PF
./kstage_check.sh hotcold4-prof "KSTAGE=hotcold KSTAGE_COLD_GB=4 KSTAGE_PROFILE=/k/p7-all.json"
echo "det done"

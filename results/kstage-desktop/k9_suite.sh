#!/bin/bash
# Is hotcold's late drift the expert permutation or a wrong host read? hc-perm0 applies the same
# profile permutation with nothing cold; if hotcold4-prof2 equals it token for token, host reads
# are exact. Then a K6 arm with per-position top 2. Runs after the k8 suite.
cd ~/kstage-tmp || exit 1
until grep -q "k8 done" k8.out 2>/dev/null || ! pgrep -f "[k]8_suite" >/dev/null; do sleep 30; done
P="KSTAGE_PROFILE=/k/p7-all.json"
./kstage_check.sh hc-perm0 "KSTAGE=hotcold KSTAGE_COLD_GB=0 $P"
./kstage_check.sh hotcold4-prof2 "KSTAGE=hotcold KSTAGE_COLD_GB=4 $P"
./kstage_check.sh cache4-s64-lp "KSTAGE=cache KSTAGE_COLD_GB=4 KSTAGE_SLOTS=64 $P"
REF=hc-perm0 python3 k9/lpcmp3.py hotcold4-prof2 hotcold4-prof
REF=hotcold4-prof2 python3 k9/lpcmp3.py hotcold4-prof
python3 k9/lpcmp3.py hc-perm0 hotcold4-prof2 cache4-s64-lp off4-lp2
echo "k9 done $(date -u +%H:%M:%SZ)"

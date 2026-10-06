#!/bin/bash
cd /home/david/moe-qlora/.claude/worktrees/next-round || exit 1
o=results/kv-levers/p18.out
echo "=== chain $(date -u +%FT%TZ): rest of p18 (the first-arm check was wrong: tail -12)" >> $o; n0=$(grep -c "" $o)
for q in p18-ks-dma64-b1-ahead-eval p18-ks-dma64-b1-ahead-k10 \
         p18-ks-dma64-b1-a8l p18-ks-dma64-b1-ahead-a8l; do
  { timeout -k 60 1800 bash probes/kv_levers.sh $q; echo "== $q exit $?"; } >> $o 2>&1
  docker rm -f kvlever-tm >/dev/null 2>&1
  if [ $q = p18-ks-dma64-b1-ahead ] && ! tail -12 $o | grep -q "^ready in"; then echo "first arm not ready: stop" >> $o; break; fi
done
if ! tail -n +$n0 $o | grep -q "first arm not ready"; then bash results/kstage-laptop/ce2_suite.sh; echo "ce2 exit $?"; fi
tail -n +$n0 $o | grep -E "^== |ready in|bench:|decode c=|agent x|exit|kstage: (cache|ahead|installed)|CE_WATCH|ce_helper|not ready" | tail -60 | cut -c1-230
grep -E "^===|ms|us|µs" results/kstage-laptop/ce2.out 2>/dev/null | tail -40 | cut -c1-200

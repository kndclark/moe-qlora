#!/bin/bash
cd /home/david/moe-qlora/.claude/worktrees/next-round || exit 1
o=results/kv-levers/p18-diag.out; d=results/kv-levers/gdb-p18-ahead.txt
( ENVS="CE_WATCH=400" timeout -k 60 900 bash probes/kv_levers.sh p18-ks-dma64-b1-ahead >> $o 2>&1; echo "== arm exit $?" >> $o ) &
sleep 30
until docker logs kvlever-tm 2>&1 | grep -q "Actual usage"; do sleep 3; [ -n "$(docker ps -q -f name=kvlever-tm)" ] || break; done
sleep 40
cp=$(docker logs kvlever-tm 2>&1 | grep -oE "EngineCore pid=[0-9]+" | head -1 | grep -oE "[0-9]+$")
hp=""; for p in $(docker top kvlever-tm -eo pid | tail -n +2); do
  [ "$(awk '/^NSpid/{print $NF}' /proc/$p/status 2>/dev/null)" = "$cp" ] && hp=$p; done
echo "EngineCore container pid $cp host pid $hp at $(date -u +%T)" > $d
docker logs kvlever-tm 2>&1 | grep "ahead helper" | tail -2 >> $d
for t in 0 15; do sleep $t; echo "=== gdb $(date -u +%T)" >> $d
  sudo -n timeout 120 gdb -p $hp -batch -ex "set pagination off" -ex "info threads" -ex "thread apply all bt 40" >> $d 2>&1; done
docker logs kvlever-tm 2>&1 | grep "ahead helper" | tail -1 >> $d
docker rm -f kvlever-tm >/dev/null 2>&1; wait
grep -c "" $d

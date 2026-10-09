#!/bin/bash
H=/home/david/handoff-0948db4b-files/hybrid
timeout 330 ssh llm 'bash ~/kt-relu2-results/hybrid/sweep.sh $HOME/kt-relu2-results/hybrid desktop' < /dev/null &
bash $H/sweep.sh $H laptop
wait
timeout 30 scp -q llm:kt-relu2-results/hybrid/desktop-h2d-sweep.jsonl $H/
for n in laptop desktop; do echo "== $n"; cat $H/$n-h2d-sweep.jsonl; tail -3 $H/$n-h2d-sweep.err; done

#!/bin/bash
# Push the files, then start uva_serve.sh on the desktop detached; output in uvathp/uva_serve-$1.out.
bash /home/david/handoff-0948db4b-files/hybrid/uva/push_desktop.sh copy || exit 1
D=kt-relu2-results/uvathp; GB=${1:-4}
timeout 20 ssh llm "cd $D && nohup bash uva_serve.sh $GB > uva_serve-$GB.out 2>&1 &" < /dev/null
sleep 5; timeout 20 ssh llm "cat $D/uva_serve-$GB.out; docker ps --format '{{.Names}} {{.Status}}'" < /dev/null

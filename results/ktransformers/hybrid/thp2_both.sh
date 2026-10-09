#!/bin/bash
# thp2.sh on both nodes at once (desktop detached), then fetch the desktop's result.
H=/home/david/handoff-0948db4b-files/hybrid
R=kt-relu2-results/hybrid
timeout 60 scp -q "$H/h2d_thp2.py" "$H/thp2.sh" "llm:$R/" || exit 1
timeout 20 ssh llm "cd $R && nohup bash thp2.sh \$HOME/$R \$HOME/$R/k desktop > desktop-thp2.log 2>&1 &" < /dev/null
echo "== laptop"
bash "$H/thp2.sh"
for i in $(seq 1 40); do
  timeout 20 ssh llm "grep -c rc= $R/desktop-thp2.log | grep -q 3" < /dev/null && break
  sleep 15
done
timeout 60 scp -q "llm:$R/desktop-h2d_thp2.*" "llm:$R/desktop-thp2.log" "$H/"
echo "== desktop"; cat "$H/desktop-thp2.log"

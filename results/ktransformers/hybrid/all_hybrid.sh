#!/bin/bash
# Desktop: VMM/THP probe then hybrid (gptq4 10 5), detached; laptop: hybrid (vnni4 16 8). Then fetch.
H=/home/david/handoff-0948db4b-files/hybrid
K=/home/david/moe-qlora/.claude/worktrees/next-round/probes/kstage
R=kt-relu2-results/hybrid
timeout 20 ssh llm "mkdir -p $R/k" < /dev/null || exit 1
timeout 60 scp -q "$K/vllm_kstage.py" "llm:$R/k/" || exit 1
timeout 60 scp -q "$H/h2d_vmm.py" "$H/vmm_desktop.sh" "$H/h2d_load.py" "$H/hybrid_run.sh" "llm:$R/" || exit 1
timeout 20 ssh llm "cd $R && nohup bash -c 'bash vmm_desktop.sh; bash hybrid_run.sh \$HOME/$R desktop gptq4 10 5' > desktop-run.log 2>&1 &" < /dev/null
echo "desktop launched"
bash "$H/hybrid_run.sh" "$H" laptop vnni4 16 8 > "$H/laptop-run.log" 2>&1
echo "laptop: $(tail -1 "$H/laptop-run.log")"
for i in $(seq 1 120); do
  timeout 20 ssh llm "grep -q hybrid_done $R/desktop-run.log" < /dev/null && break
  sleep 30
done
timeout 120 scp -q "llm:$R/desktop-*" "$H/"
echo "== desktop log"; cat "$H/desktop-run.log"
echo "== desktop vmm"; cat "$H/desktop-h2d_vmm.jsonl"; grep -v Warning "$H/desktop-h2d_vmm.err" | tail -3
echo "== laptop log"; cat "$H/laptop-run.log"
ls "$H" | grep -E "hybrid|solo|during" | tr '\n' ' '

#!/bin/bash
# Copy kstage + the UVA test to the desktop scratch dir (created by this job) and run the test there.
D=kt-relu2-results/uvathp
U=/home/david/handoff-0948db4b-files/hybrid/uva
K=/home/david/moe-qlora/.claude/worktrees/next-round/probes
timeout 20 ssh llm "mkdir -p $D/k" < /dev/null || exit 1
timeout 120 scp -q -r "$K/kstage/vllm_kstage.py" "$K/kstage/vllm_kstage-0.1.dist-info" "llm:$D/k/" || exit 1
timeout 60 scp -q "$K/offload_bench.py" "$U/uva_test.py" "$U/uva_test.sh" "$U"/uva_*.sh "llm:$D/" || exit 1
if [ "${1:-test}" = test ]; then timeout 700 ssh llm "cd $D && bash uva_test.sh \$HOME/$D \$HOME/$D/k" < /dev/null; fi

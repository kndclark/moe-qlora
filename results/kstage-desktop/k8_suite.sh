#!/bin/bash
# Prefetch race fix (KSTAGE_PFFIX) and per-token logprobs, after the predict suite. The base,
# step 1 and UVA reruns carry each position's top 2, so late divergences can be read as ties.
cd ~/kstage-tmp || exit 1
until grep -q "predict done" k7.out 2>/dev/null || ! pgrep -f "[k]7_suite" >/dev/null; do sleep 30; done
cp k8/vllm_kstage.py vllm_kstage.py; cp k8/kstage_client.py kstage_client.py
echo "== pffix_test $(date -u +%H:%M:%SZ)"
docker run --rm -v "$PWD":/k:ro -w /k/k8 --entrypoint python3 --pull never vllm/vllm-openai:v0.29.0 \
  /k/k8/pffix_test.py 2>&1 | tail -2
PF="--offload-backend prefetch --offload-params experts --offload-group-size 4"
./kstage_check.sh base-lp2 ""
./kstage_check.sh pf4-s2-fix "KSTAGE_PFFIX=1" $PF --offload-num-in-group 1 --offload-prefetch-step 2
./kstage_check.sh pf4-r2b "" $PF --offload-num-in-group 1
./kstage_check.sh pf4k2-s2 "" $PF --offload-num-in-group 2 --offload-prefetch-step 2
./kstage_check.sh off4-lp2 "" --cpu-offload-params experts --cpu-offload-gb 4
python3 k8/lpcmp2.py pf4-s2-fix pf4-r2b pf4k2-s2 off4-lp2
echo "k8 done $(date -u +%H:%M:%SZ)"

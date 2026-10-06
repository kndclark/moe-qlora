#!/bin/bash
# The next-layer prediction probe on the desktop, after the K6 suite: CPU test, then one server.
cd ~/kstage-tmp || exit 1
until grep -qE "k6 (done|stopped)" k6.out 2>/dev/null; do sleep 30; done
cp k7/vllm_kstage.py vllm_kstage.py
echo "== predict_test $(date -u +%H:%M:%SZ)"
docker run --rm -v "$PWD":/k:ro -w /k/k7 --entrypoint python3 --pull never vllm/vllm-openai:v0.29.0 \
  /k/k7/predict_test.py 2>&1 | tail -1
./kstage_check.sh predict "KSTAGE=predict KSTAGE_COLD_GB=4 KSTAGE_PROFILE=/k/p7-all.json KSTAGE_STATS=20"
grep -E "kstage: predict" out/serve-predict.log | tail -12
echo "predict done $(date -u +%H:%M:%SZ)"

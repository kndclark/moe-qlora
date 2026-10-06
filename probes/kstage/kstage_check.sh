#!/bin/bash
# kstage correctness and speed on one GPU: one server per config, greedy outputs + speed.
# usage: kstage_check.sh TAG "ENV=.. ENV=.." [serve flags...]   (run from the kstage dir)
# Lightning NVFP4 without the adapter; --linear-backend marlin for the 3090 (sm_86).
# KC_DOCKER adds docker args (an adapter mount), KC_MODEL names the model the client asks for,
# KC_LINEAR replaces marlin (auto on the 5090).
set -u
tag=$1; envs=$2; shift 2
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$here/out"
name=kstage-check; B=http://127.0.0.1:8313
envx=(); for e in $envs; do envx+=(-e "$e"); done
dx=(); for e in ${KC_DOCKER:-}; do dx+=("$e"); done
docker rm -f $name >/dev/null 2>&1
echo "== $tag $(date -u +%H:%M:%SZ) env: ${envs:-none} flags: $*"
docker run -d --name $name --gpus all --ipc=host -p 127.0.0.1:8313:8000 \
  -v /srv/model-cache:/hf:ro -v "$here":/k:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/k "${envx[@]}" "${dx[@]}" \
  --pull never vllm/vllm-openai:v0.29.0 \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --revision bee7596271d1495f6992ae224aefde4410e816b8 --served-model-name lightning-nvfp4 \
  --kv-cache-dtype fp8 --mamba-cache-mode align --moe-backend marlin --linear-backend "${KC_LINEAR:-marlin}" \
  --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.85 "$@" >/dev/null || exit 1
t0=$(date +%s)
until curl -sf $B/health >/dev/null; do
  if [ "$(docker inspect -f '{{.State.Running}}' $name 2>/dev/null)" != true ] || [ $(( $(date +%s)-t0 )) -gt 900 ]; then
    echo "not ready ($(( $(date +%s)-t0 ))s)"; docker logs $name > "$here/out/serve-$tag.log" 2>&1
    grep -E "kstage:|Error|error" "$here/out/serve-$tag.log" | tail -15; docker rm -f $name >/dev/null; exit 2; fi
  sleep 5
done
docker logs $name > "$here/out/serve-$tag.log" 2>&1
echo "ready in $(( $(date +%s)-t0 ))s; $(grep -oE "Model loading took [0-9.]+ GiB|Available KV cache memory: [0-9.]+ GiB|Total CPU offloaded parameters: [0-9.]+" "$here/out/serve-$tag.log" | tr '\n' ';')"
grep -E "kstage:|PrefetchOffloader\] Init" "$here/out/serve-$tag.log" | sed -E "s/^.*(kstage:|\[PrefetchOffloader)/  \1/" | head -40
python3 "$here/kstage_client.py" $B "${KC_MODEL:-lightning-nvfp4}" "$here/out/$tag.json"
docker logs $name > "$here/out/serve-$tag.log" 2>&1
docker rm -f $name >/dev/null

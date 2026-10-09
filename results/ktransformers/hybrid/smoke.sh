#!/bin/bash
D=/home/david/handoff-0948db4b-files/hybrid
timeout 120 docker run --rm --name h2d-smoke --pull never --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
  --entrypoint python3 -v "$D:/w" vllm/vllm-openai:v0.29.0 -u /w/h2d_load.py --seconds 5 --out /w/smoke.jsonl 2>&1 | tail -5
head -2 $D/smoke.jsonl; tail -3 $D/smoke.jsonl

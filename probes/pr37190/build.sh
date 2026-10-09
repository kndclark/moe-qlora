#!/usr/bin/env bash
# Builds vllm-openai:v0.29.0-pr37190 locally from vllm/vllm-openai:v0.29.0. Never push it.
set -eu
cd "$(dirname "$0")"
docker build --pull=false -t vllm-openai:v0.29.0-pr37190 .

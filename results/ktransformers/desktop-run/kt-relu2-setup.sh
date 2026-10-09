#!/bin/bash
# Runs ON the desktop: kt-relu2 container (vLLM image + hwloc/cmake), AVX2 build with 10 jobs, relu2 test
set -uo pipefail
SRC=$HOME/kt-relu2-src
docker run -d --name kt-relu2 --pull never --entrypoint sleep -v "$SRC":/ws:ro vllm/vllm-openai:v0.29.0 infinity || exit 1
docker exec kt-relu2 bash -c 'set -e; apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libhwloc-dev pkg-config git >/dev/null && pip install -q cmake pybind11 wheel && (python3 -m pytest --version || pip install -q pytest)' || exit 1
docker exec kt-relu2 bash -c 'rm -rf /src && mkdir -p /src/third_party && cp -a /ws/kt-kernel /src/ && cp -a /ws/third_party/llama.cpp /ws/third_party/pybind11 /ws/third_party/llamafile /src/third_party/ && cd /src/kt-kernel && CPUINFER_CPU_INSTRUCT=AVX2 CPUINFER_ENABLE_AMX=OFF CPUINFER_USE_CUDA=0 CPUINFER_BUILD_TYPE=Release CPUINFER_PARALLEL=10 pip install --no-deps --no-build-isolation -v . > /src/build.log 2>&1; rc=$?; tail -3 /src/build.log; echo build_rc=$rc; pip install -q --no-deps /src/third_party/llama.cpp/gguf-py'
docker exec kt-relu2 bash -c 'cd /src/kt-kernel/test && python3 -m pytest -q -p no:cacheprovider per_commit/test_moe_avx2_relu2.py > /src/test.log 2>&1; echo pytest_rc=$?; tail -2 /src/test.log'
echo setup_done

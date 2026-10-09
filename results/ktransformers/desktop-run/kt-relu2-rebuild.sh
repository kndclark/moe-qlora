#!/bin/bash
# Runs ON the desktop: rebuild kt-kernel in the existing kt-relu2 container, then the relu2 and upstream GPTQ tests
set -uo pipefail
docker exec kt-relu2 bash -c 'rm -rf /src && mkdir -p /src/third_party && cp -a /ws/kt-kernel /src/ && cp -a /ws/third_party/llama.cpp /ws/third_party/pybind11 /ws/third_party/llamafile /src/third_party/ && cd /src/kt-kernel && CPUINFER_CPU_INSTRUCT=AVX2 CPUINFER_ENABLE_AMX=OFF CPUINFER_USE_CUDA=0 CPUINFER_BUILD_TYPE=Release CPUINFER_PARALLEL=10 pip install --no-deps --no-build-isolation -v . > /src/build.log 2>&1; rc=$?; grep -nE " error|error:" /src/build.log | head -20; tail -2 /src/build.log; echo build_rc=$rc'
docker exec kt-relu2 bash -c 'cd /src/kt-kernel/test && python3 -m pytest -v -s -p no:cacheprovider per_commit/test_moe_avx2_relu2.py per_commit/test_moe_gptq_int4_accuracy.py > /src/test.log 2>&1; rc=$?; grep -aE "PASSED|FAILED|SKIPPED|ERROR|passed|failed|diff=" /src/test.log | cut -c1-200 | tail -70; echo pytest_rc=$rc'
echo rebuild_done

#!/usr/bin/env bash
# Builds llama.cpp's llama-server and llama-quantize with CUDA inside vllm/vllm-openai:v0.29.0,
# which carries nvcc 13.0 and cuBLAS; neither node has a CUDA toolkit of its own. Targets are the
# 3090 (86) and the laptop's card (120a, the Blackwell target with FP4 tensor cores), both -real
# so no PTX is embedded to be compiled again at load. The binaries run in the same image.
# 12 jobs, not ninja's 26: an nvcc job on the fattn/mmq templates can take several GiB of RAM.
# Usage: probes/llamacpp/build.sh [SRC]   SRC = a llama.cpp checkout (default ~/llama.cpp-src/llama.cpp)
set -eu
src=$(cd "${1:-$HOME/llama.cpp-src/llama.cpp}" && pwd)
docker run --rm --pull never --entrypoint sh -v "$src":/src -w /src -e HOME=/tmp \
  --user "$(id -u):$(id -g)" vllm/vllm-openai:v0.29.0 -c '
  pip install -q --user cmake && export PATH=/tmp/.local/bin:$PATH &&
  cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="86-real;120a-real" -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF &&
  cmake --build build -j 12 --target llama-server llama-quantize'

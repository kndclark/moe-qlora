#!/bin/bash
# Usage: run_all.sh RUN [LABEL...]   e.g. run_all.sh r2 fi-simple fi-cta triton
# Serves each config in turn (default order: triton fi-cta fi-simple) via serve_ab.sh;
# outputs land in results/ssu_sm120/serve/RUN/LABEL. fi-cta bind-mounts ../patch/include.
set -u
H=$(cd "$(dirname "$0")" && pwd); P=$(cd "$H/../patch/include" && pwd)
I=/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/mamba
export RUN=$1; shift; [ $# -gt 0 ] || set -- triton fi-cta fi-simple
m=(); for f in stp mtp_vertical mtp_horizontal; do
  m+=(-v "$P/kernel_selective_state_update_$f.cuh:$I/kernel_selective_state_update_$f.cuh:ro"); done
for L in "$@"; do
  case $L in
    triton) "$H/serve_ab.sh" triton ;;
    fi-cta) "$H/serve_ab.sh" fi-cta --docker "${m[@]}" --vllm --mamba-backend flashinfer ;;
    fi-simple) "$H/serve_ab.sh" fi-simple --vllm --mamba-backend flashinfer --mamba-ssu-algorithm simple ;;
    *) echo "unknown label: $L"; exit 1 ;;
  esac
done
echo ALL_DONE

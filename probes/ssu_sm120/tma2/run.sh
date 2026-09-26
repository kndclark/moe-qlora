#!/bin/bash
# Run inside the image with this dir at /w (no GPU needed):
#   docker run --rm -v "$PWD":/w --entrypoint bash vllm/vllm-openai:v0.29.0 /w/run.sh
cd /w
for NV in /usr/local/cuda-13.0/bin/nvcc /usr/local/lib/python3.12/dist-packages/nvidia/cu13/bin/nvcc; do
  echo "#### $NV: $($NV --version | grep release)"
  for A in sm_120a sm_100a; do
    for V in "" "-DCTA"; do
      o=f_${A}${V}.cubin
      if $NV -arch=$A $V -cubin -o $o forms2.cu 2> err.txt; then
        echo "== $A ${V:-cluster-only}: $(cuobjdump -symbols $o 2>/dev/null | grep -c __cuda_syscall) syscall refs"
        cuobjdump -res-usage $o 2>/dev/null | grep -E "Function|REG" | paste - - | sed -E 's/.*Function ([a-z_]+).*REG:([0-9]+) STACK:([0-9]+).*/   \1 REG:\2 STACK:\3/'
      else echo "== $A ${V:-cluster-only}: COMPILE FAIL: $(head -c 300 err.txt)"; fi
    done
  done
done
for P in $(find / -path /proc -prune -o -name ptxas -type f -print 2>/dev/null); do echo "ptxas: $P $($P --version | grep release)"; done

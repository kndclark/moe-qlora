#!/bin/bash
docker cp poison.c kt-relu2:/tmp/poison.c && docker cp kt_poison.py kt-relu2:/tmp/kt_poison.py
docker exec kt-relu2 bash -c 'gcc -shared -fPIC -O1 -o /tmp/poison.so /tmp/poison.c -ldl && grep -rn "numa_alloc\|posix_memalign" /src0/kt-kernel/cpu_backend/shared_mem_buffer.cpp | head -4'
for b in src0 src; do for pb in 0x7f 0x00; do
  P=/$b/kt-kernel/build/lib.linux-x86_64-cpython-312/kt_kernel
  docker exec -e KT_KERNEL_EXT_DIR=$P -e LD_PRELOAD=/tmp/poison.so -e POISON_BYTE=$pb kt-relu2 bash -c "cd /$b/kt-kernel/test && timeout 600 python3 /tmp/kt_poison.py /$b/kt-kernel/test > /tmp/poison-$b-$pb.log 2>&1; echo $b $pb rc=\$?: poisoned=\$(grep -c '^\[poison\]' /tmp/poison-$b-$pb.log) \$(grep -a 'diff =\|NONALIGNED\|Error' /tmp/poison-$b-$pb.log | tr -s ' ' | tr '\n' ' ' | cut -c1-260)"
done; done

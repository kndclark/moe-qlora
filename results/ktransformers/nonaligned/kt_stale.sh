#!/bin/bash
for b in src0 src; do
  P=/$b/kt-kernel/build/lib.linux-x86_64-cpython-312
  P=$P/kt_kernel
  docker exec -e KT_KERNEL_EXT_DIR=$P kt-relu2 bash -c "cd /$b/kt-kernel/test && timeout 900 python3 /tmp/kt_stale.py /$b/kt-kernel/test > /tmp/stale-$b.log 2>&1; echo $b rc=\$?: \$(grep -a 'diff =\|NONALIGNED\|Error' /tmp/stale-$b.log | tr -s ' ' | tr '\n' ' ' | cut -c1-300)"
done

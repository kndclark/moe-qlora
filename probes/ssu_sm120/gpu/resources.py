"""Kernel + device resource numbers via the driver API (cuda.bindings).
Loads the extracted cubin with cuModuleLoad, queries cuFuncGetAttribute for
the STP kernels, and prints device limits. Usage: resources.py <cubin>..."""
import json
import sys

from cuda.bindings import driver as cu


def ok(r):
    if not isinstance(r, tuple):
        r = (r,)
    err = r[0]
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(str(err))
    rest = r[1:]
    return rest[0] if len(rest) == 1 else (rest or None)


ok(cu.cuInit(0))
dev = ok(cu.cuDeviceGet(0))
ctx = ok(cu.cuDevicePrimaryCtxRetain(dev))
ok(cu.cuCtxSetCurrent(ctx))
A = cu.CUdevice_attribute
devattrs = {}
for name in ["CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT",
             "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR",
             "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
             "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",
             "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
             "CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK",
             "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR",
             "CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR",
             "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR",
             "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR",
             "CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH"]:
    devattrs[name.replace("CU_DEVICE_ATTRIBUTE_", "")] = ok(cu.cuDeviceGetAttribute(getattr(A, name), dev))
free, total = ok(cu.cuMemGetInfo())
devattrs["total_mem_MiB"] = total / 2**20
devattrs["stack_limit_B"] = ok(cu.cuCtxGetLimit(cu.CUlimit.CU_LIMIT_STACK_SIZE))
print("DEVICE " + json.dumps(devattrs))

F = cu.CUfunction_attribute
want = ["producer_consumer_vertical", "producer_consumer_horizontal", "kernel_simpleI"]
for path in sys.argv[1:]:
    mod = ok(cu.cuModuleLoad(path.encode()))
    n = ok(cu.cuModuleGetFunctionCount(mod))
    funcs = ok(cu.cuModuleEnumerateFunctions(n, mod))
    for f in funcs:
        name = ok(cu.cuFuncGetName(f))
        name = name.decode() if isinstance(name, bytes) else str(name)
        if "mtp" in name or not any(w in name for w in want):
            continue
        ok(cu.cuFuncLoad(f))  # lazy loading: materialise before querying
        rec = {"cubin": path.rsplit(".", 3)[-3], "name": name[:140]}
        for a in ["LOCAL_SIZE_BYTES", "SHARED_SIZE_BYTES", "NUM_REGS",
                  "MAX_DYNAMIC_SHARED_SIZE_BYTES", "MAX_THREADS_PER_BLOCK",
                  "BINARY_VERSION", "PTX_VERSION", "CONST_SIZE_BYTES"]:
            rec[a] = ok(cu.cuFuncGetAttribute(getattr(F, "CU_FUNC_ATTRIBUTE_" + a), f))
        print("FUNC " + json.dumps(rec))
devattrs2 = ok(cu.cuCtxGetLimit(cu.CUlimit.CU_LIMIT_STACK_SIZE))
print("STACK_LIMIT_AFTER_LOAD " + str(devattrs2))

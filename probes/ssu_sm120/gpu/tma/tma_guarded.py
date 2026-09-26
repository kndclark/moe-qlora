import json, subprocess, cupy as cp
subprocess.run(["nvcc", "-cubin", "-arch=sm_120a", "-o", "/work/tma/tma_guarded.cubin", "/work/tma/tma_guarded.cu"], check=True)
rt = cp.cuda.runtime
mod = cp.RawModule(path="/work/tma/tma_guarded.cubin")
out = cp.zeros(128, dtype=cp.float32); dummy = cp.zeros(32, dtype=cp.uint32)
def st(): cp.cuda.Device().synchronize(); return dict(free_mib=rt.memGetInfo()[0] / 2**20, stack_B=rt.deviceGetLimit(0))
r = {"start": st()}
for name in ["no_tma", "tma2d_guarded"]:
    k = mod.get_function(name)
    k((82,), (128,), (dummy, cp.int32(0), out))
    r[name] = st(); r[name]["local_size_bytes"] = k.attributes["local_size_bytes"]; r[name]["out_ok"] = bool((out[:82] == 1).all())
print("RESULT " + json.dumps(r))

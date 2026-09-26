"""Run probe_launch.py once per config, each in a fresh process (fresh CUDA
context, so every run sees a genuine first launch)."""
import json
import subprocess
import sys

configs = [a.split(",") for a in sys.argv[1:]]  # "algo,batch[,leave_free_mib]"
for c in configs:
    cmd = [sys.executable, "/work/probe_launch.py", "--algo", c[0], "--batch", c[1]]
    if len(c) > 2:
        cmd += ["--leave-free-mib", c[2]]
    p = subprocess.run(cmd, capture_output=True, text=True)
    res = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    errs = [l for l in p.stderr.splitlines()
            if "cuCtxGetDevice" not in l and "No CUDA context" not in l and l.strip()]
    r = json.loads(res[0][7:]) if res else {"cfg": c, "rc": p.returncode}
    r["driver_log"] = [e[:300] for e in errs if "[CUDA]" in e][:6]
    if not res:
        r["stderr_tail"] = [e[:300] for e in errs[-8:]]
    print("ROW " + json.dumps(r), flush=True)

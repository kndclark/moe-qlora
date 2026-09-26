"""Run probe_stack.py per config in fresh processes. Config: kernel[,stack_limit[,local_bytes]]"""
import json, subprocess, sys
for a in sys.argv[1:]:
    c = a.split(",")
    cmd = [sys.executable, "/work/probe_stack.py", "--kernel", c[0]]
    if len(c) > 1 and c[1]:
        cmd += ["--stack-limit", c[1]]
    if len(c) > 2:
        cmd += ["--local-bytes", c[2]]
    p = subprocess.run(cmd, capture_output=True, text=True)
    res = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if res:
        print("ROW " + res[0][7:], flush=True)
    else:
        print("ROW " + json.dumps({"cfg": a, "rc": p.returncode, "tail": p.stderr.splitlines()[-6:]}), flush=True)

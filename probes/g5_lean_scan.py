"""G5 option 2: g5_train_step.py, byte for byte, with lean_scan installed first.

A new configuration (Route 1 + lean mamba scan), approved by David 2026-09-24. It is
recorded beside the Route 1-as-run gate result, never in place of it. Same pass lines.
After the probe exits, its JSON gains a "lean_scan" block: whether the patch was still
installed and how many scan calls went through it (0 would mean the patch did nothing).

Run exactly as g5_train_step.py, with this file as the entrypoint script:
  ... --entrypoint python3 gpu-lab:training /probes/g5_lean_scan.py LABEL [512,1024,2048]
"""
import json
import os
import runpy
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import lean_scan  # noqa: E402

lean_scan.install()
label = sys.argv[1]
sys.argv = [os.path.join(here, "g5_train_step.py")] + sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    path = f"/out/{label}.json"
    if os.path.exists(path):
        res = json.load(open(path))
        res["lean_scan"] = lean_scan.report()
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
    print("lean_scan", json.dumps(lean_scan.report()))

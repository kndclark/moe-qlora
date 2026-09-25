"""G2 with the lean mamba scan: g2_fidelity.py, byte for byte, with lean_scan installed first.

David asked for this run 2026-09-25, after the lean scan passed G5. Rounding alone moves
Lightning's step-0 loss by up to ~0.03 (lean-scan-*-check-desktop.json), so G2 is recorded
under both scans, never one in place of the other. The as-defined gate result is
g2_fidelity.py run plainly; this is the Route 1 + lean scan configuration. Same pass lines.

The patch replaces nh.mamba2_chunk_scan for the whole process, so the bf16 reference
(phase A) and Route 1 (phase B) both run the lean scan: the comparison is 4-bit damage
under the lean scan. Expected scan calls (ARITHMETIC): 23 mamba layers x 4 windows x
3 passes (reference, accumulated, isolated) = 276. Qwen3-8B has no mamba layers, so its
yardstick needs no rerun; lightning mode only.

After the probe exits, its JSON gains a "lean_scan" block (installed, calls).
  ... --entrypoint python3 gpu-lab:training /probes/g2_lean_scan.py lightning LABEL
"""
import json
import os
import runpy
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import lean_scan  # noqa: E402

assert sys.argv[1] == "lightning", "lean scan G2 is lightning mode only"
lean_scan.install()
label = sys.argv[2]
sys.argv = [os.path.join(here, "g2_fidelity.py")] + sys.argv[1:]
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

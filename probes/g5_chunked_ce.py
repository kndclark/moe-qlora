"""F1 (plan.md): g5_train_step.py, byte for byte, with lean_scan and chunked_ce installed.

David approved trying chunked cross-entropy 2026-09-25 to fit seq 2048 on the laptop;
g5_mem_phases.py found the stock step's seq-2048 peak in the loss backward. Recorded beside
the G5 results, same pass lines. After the probe exits, its JSON gains "lean_scan" and
"chunked_ce" blocks (calls = 0 would mean the patch did nothing). CE_CHUNK sets the chunk
(default 256 positions).

For the verdict configuration, run it through the attention-bf16 wrapper:
  ... --entrypoint python3 gpu-lab:training /probes/attn_bf16.py g5_chunked_ce.py LABEL [512,1024,2048]
"""
import json
import os
import runpy
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import chunked_ce  # noqa: E402
import lean_scan  # noqa: E402

lean_scan.install()
chunked_ce.install(int(os.environ.get("CE_CHUNK", "256")))
label = sys.argv[1]
sys.argv = [os.path.join(here, "g5_train_step.py")] + sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    path = f"/out/{label}.json"
    if os.path.exists(path):
        res = json.load(open(path))
        res["lean_scan"] = lean_scan.report()
        res["chunked_ce"] = chunked_ce.report()
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
    print("lean_scan", json.dumps(lean_scan.report()), "chunked_ce", json.dumps(chunked_ce.report()))

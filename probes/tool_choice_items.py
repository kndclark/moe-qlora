"""Per-item tool choice on the alert and promqlcat sets, thinking on: for each run label,
items correct, run status, calls by tool name, turns that open with an empty "</think>",
and per item whether it was correct and which tools it called in order.

usage: python3 probes/tool_choice_items.py OUT.json LABEL [LABEL...]
  LABEL "base" reads research-eval-<set>-lightning-think-4k.json, any other
  research-eval-<set>-lightning-<LABEL>-think-4k.json, both from results/.
"""
import collections
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path, labels = sys.argv[1], sys.argv[2:]
report = {}
for s in ("alert", "promqlcat"):
    runs = {}
    for lab in labels:
        tag = "" if lab == "base" else f"-{lab}"
        d = json.load(open(os.path.join(REPO, "results", f"research-eval-{s}-lightning{tag}-think-4k.json")))
        runs[lab] = {r["id"]: r for r in d["results"]}
    summary = {}
    for lab, rs in runs.items():
        turns = [t for r in rs.values() for t in r["run"]["turns"]]
        summary[lab] = {
            "correct": sum(bool(r["score"].get("correct")) for r in rs.values()), "items": len(rs),
            "status": dict(collections.Counter(r["run"]["status"] for r in rs.values())),
            "calls": dict(collections.Counter(c.get("name") for r in rs.values() for c in r["run"]["calls"])),
            "zero_call_items": sum(not r["run"]["calls"] for r in rs.values()),
            "empty_think_turns": sum(t["text"].lstrip().startswith("</think>") for t in turns), "turns": len(turns)}
    items = {i: {lab: {"correct": bool(runs[lab][i]["score"].get("correct")),
                       "tools": [c.get("name") for c in runs[lab][i]["run"]["calls"]]} for lab in runs}
             for i in runs[labels[0]]}
    report[s] = {"summary": summary, "items": items}
    print(f"== {s}")
    for lab, v in summary.items():
        print(f"  {lab:10} correct {v['correct']}/{v['items']}  status {v['status']}  calls {v['calls']}  "
              f"zero-call items {v['zero_call_items']}  empty-think turns {v['empty_think_turns']}/{v['turns']}")
json.dump(report, open(out_path, "w"), indent=1)

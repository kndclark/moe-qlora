"""Thinking-on turns per eval set: how many close </think> with non-empty reasoning, with
empty reasoning, or never close it (the think trap), plus run statuses and finish reasons.

usage: python3 probes/think_counts.py OUT.json LABEL [LABEL...]
  reads results/research-eval-<set>-lightning-<LABEL>-think-4k.json
"""
import collections
import glob
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path, labels = sys.argv[1], sys.argv[2:]
report = {}
for lab in labels:
    per = collections.defaultdict(lambda: {"turns": 0, "reasoned": 0, "empty": 0, "no_close": 0})
    status, finish, chars = collections.Counter(), collections.Counter(), []
    for f in sorted(glob.glob(os.path.join(REPO, "results", f"research-eval-*-lightning-{lab}-think-4k.json"))):
        s = os.path.basename(f).split("research-eval-")[1].split("-lightning")[0]
        for r in json.load(open(f))["results"]:
            status[r["run"]["status"]] += 1
            for t in r["run"]["turns"]:
                if "text" not in t:
                    continue
                tx = t["text"]
                per[s]["turns"] += 1
                finish[t.get("finish_reason")] += 1
                if "</think>" not in tx:
                    per[s]["no_close"] += 1
                elif tx.split("</think>")[0].strip():
                    per[s]["reasoned"] += 1
                    chars.append(len(tx.split("</think>")[0].strip()))
                else:
                    per[s]["empty"] += 1
    tot = {k: sum(v[k] for v in per.values()) for k in ("turns", "reasoned", "empty", "no_close")}
    report[lab] = {"total": tot, "by_set": dict(per), "status": dict(status), "finish": dict(finish),
                   "reasoning_chars_median": statistics.median(chars) if chars else 0}
    print(f"{lab}: {tot}, median reasoning chars {report[lab]['reasoning_chars_median']}")
    print("   " + ", ".join(f"{s} {v['reasoned']}/{v['turns']}" for s, v in per.items()))
json.dump(report, open(out_path, "w"), indent=1)

"""Trap rows, thinking on, pooled over repeats: how each item fails. pass = denied
(trap, trap2, rocky_trap) or noticed (trap3); else the run status when it is not
"answered" (call_limit, truncated_in_think, ...), else fabricated (trap3), else answered
without denying.

usage: python3 probes/trap_failures.py OUT.json "r1 r2 r3" LABEL [LABEL...]
  r1 = the gate's run in results/, others in results/noise/ (probes/noise_eval.sh).
"""
import collections
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = (("v1", "trap"), ("v2", "trap2"), ("rocky", "rocky_trap"), ("trap3", "trap3"))


def path(tag, label, rep):
    lab = "" if label == "base" else f"-{label}"
    if rep == "r1":
        return os.path.join(REPO, "results", f"research-eval-{tag}-lightning{lab}-think-4k.json")
    return os.path.join(REPO, "results", "noise", f"research-eval-{tag}-lightning{lab}-think-4k-{rep}.json")


out_path, reps, labels = sys.argv[1], sys.argv[2].split(), sys.argv[3:]
report = {}
for lab in labels:
    for tag, split in ROWS:
        c = collections.Counter()
        used = []
        for rep in reps:
            p = path(tag, lab, rep)
            if not os.path.exists(p):
                continue
            used.append(rep)
            for r in json.load(open(p))["results"]:
                if r["split"] != split:
                    continue
                s, st = r["score"], r["run"]["status"]
                if (s.get("noticed") if split == "trap3" else s.get("denied_heuristic")):
                    c["pass"] += 1
                elif st != "answered":
                    c[st] += 1
                elif split == "trap3" and s.get("fabricated"):
                    c["fabricated"] += 1
                else:
                    c["answered_not_denied"] += 1
        report[f"{lab}/{split}"] = {"repeats": used, **dict(sorted(c.items()))}
        print(f"{lab:5} {split:11} repeats {','.join(used):9} {dict(sorted(c.items()))}")
json.dump(report, open(out_path, "w"), indent=1)

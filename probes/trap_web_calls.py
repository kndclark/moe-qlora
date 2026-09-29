"""G6w's mechanism read (plan.md "G6w"): how often each model calls web_search, thinking on,
pooled over repeats, and what happens on the trap items where it does.

Per label: every item of the thinking-on runs (v1 trap rows only in r1, as N1), items that
call web_search and the calls made; then the trap rows (trap, trap2, rocky_trap, trap3; pass
as probes/trap_failures.py) split by whether the item called web_search, and each failure
after a web_search by kind: the run status when it is not "answered" (truncated_in_think,
call_limit, ...), else fabricated (trap3), else answered without denying.

usage: python3 probes/trap_web_calls.py OUT.json "r1 r2 r3" LABEL [LABEL...]
  r1 = the gate's run in results/, others in results/noise/ (probes/noise_eval.sh).
"""
import collections
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = ("v1", "v2", "rocky", "promqlcat", "alert", "trap3")
TRAPS = ("trap", "trap2", "rocky_trap", "trap3")


def path(tag, label, rep):
    lab = "" if label == "base" else f"-{label}"
    if rep == "r1":
        return os.path.join(REPO, "results", f"research-eval-{tag}-lightning{lab}-think-4k.json")
    return os.path.join(REPO, "results", "noise", f"research-eval-{tag}-lightning{lab}-think-4k-{rep}.json")


def kind(r):
    s, st, split = r["score"], r["run"]["status"], r["split"]
    if s.get("noticed") if split == "trap3" else s.get("denied_heuristic"):
        return "pass"
    if st != "answered":
        return st
    if split == "trap3" and s.get("fabricated"):
        return "fabricated"
    return "answered_not_denied"


out_path, reps, labels = sys.argv[1], sys.argv[2].split(), sys.argv[3:]
report = {}
for lab in labels:
    allc, trap = collections.Counter(), collections.Counter()
    after = collections.Counter()
    used = collections.defaultdict(list)
    for tag in TAGS:
        for rep in reps:
            p = path(tag, lab, rep)
            if not os.path.exists(p):
                continue
            used[tag].append(rep)
            for r in json.load(open(p))["results"]:
                ws = sum(c["name"] == "web_search" for c in r["run"]["calls"])
                allc["items"] += 1
                allc["web_items"] += ws > 0
                allc["web_calls"] += ws
                if r["split"] not in TRAPS:
                    continue
                k = kind(r)
                w = "web" if ws else "no_web"
                trap[f"{w}_items"] += 1
                trap[f"{w}_fail"] += k != "pass"
                if ws and k != "pass":
                    after[k] += 1
    report[lab] = {"repeats": dict(used), "all_items": dict(allc), "trap": dict(sorted(trap.items())),
                   "trap_fail_after_web": dict(sorted(after.items()))}
    print(f"{lab:5} all {dict(allc)}\n      trap {dict(sorted(trap.items()))}\n"
          f"      trap failures after web_search {dict(sorted(after.items()))}")
json.dump(report, open(out_path, "w"), indent=1)

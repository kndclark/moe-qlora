"""N1 (plan.md "N1"): run-to-run spread of the thinking-on headline rows, per model.

Repeat r1 is the gate's own run (results/research-eval-<tag>-lightning[-LABEL]-think-4k.json),
r2, r3, ... are probes/noise_eval.sh's (results/noise/...-think-4k-rN.json). Per model
and split: the headline metric in items (round(value x n), as g6_compare counts), its
mean and range over the repeats, and item stability (share of items with the same
outcome in every repeat). Then each pair of models: the difference of means, against the
larger of the two ranges.

usage: python3 probes/noise_summary.py OUT.json "r2 r3" base g6q g6t g6u
"""
import itertools
import json
import os
import statistics
import sys

here = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(here)
# g6_compare.py's headline metrics, copied: importing it would run its report.
KIND = {"held_out": "flag", "seen_tool": "flag", "trap": "trap", "trap_control": "trap",
        "no_tool": "no_tool", "held_out2": "flag", "two_flag": "flag", "fix_cmd": "flag",
        "task": "flag", "trap2": "trap", "trap2_control": "trap", "rocky_held_out": "flag",
        "rocky_task": "flag", "rocky_trap": "trap", "rocky_trap_control": "trap",
        "promql": "live", "general": "no_tool", "alert": "alert", "trap3": "trap3"}


def headline(split):
    kind = KIND[split]
    if kind == "flag":
        return [("hit_and_grounded", 1)]
    if kind == "trap":
        return [("denied_heuristic", -1 if split.endswith("_control") else 1)]
    if kind == "no_tool":
        return [("over_trigger", -1), ("correct_where_scorable", 1)]
    if kind in ("live", "alert"):
        return [("correct", 1)]
    return [("noticed", 1), ("fabricated", -1)]
TAGS = ["v2", "rocky", "promqlcat", "alert", "trap3"]


def path(tag, label, rep):
    lab = "" if label == "base" else f"-{label}"
    if rep == "r1":
        return os.path.join(REPO, "results", f"research-eval-{tag}-lightning{lab}-think-4k.json")
    return os.path.join(REPO, "results", "noise", f"research-eval-{tag}-lightning{lab}-think-4k-{rep}.json")


def main():
    out_path, reps, labels = sys.argv[1], ["r1"] + sys.argv[2].split(), sys.argv[3:]
    rows = {}
    for label in labels:
        for tag in TAGS:
            runs = [json.load(open(path(tag, label, r))) for r in reps if os.path.exists(path(tag, label, r))]
            if len(runs) < len(reps):
                print(f"{label} {tag}: {len(runs)} of {len(reps)} repeats present")
            for split in runs[0]["summary"]:
                for metric, sign in headline(split):
                    vals, per_item = [], {}
                    for run in runs:
                        s = run["summary"][split]
                        v = s.get(metric)
                        vals.append(None if v is None else round(v * s["n"]))
                        for r in run["results"]:
                            if r["split"] == split:
                                per_item.setdefault(r["id"], []).append(bool(r["score"].get(metric)))
                    ok = [v for v in vals if v is not None]
                    stable = sum(len(set(x)) == 1 for x in per_item.values())
                    rows[(label, split, metric)] = {
                        "n": runs[0]["summary"][split]["n"], "items": vals, "sign": sign,
                        "mean": round(statistics.mean(ok), 2) if ok else None,
                        "range": max(ok) - min(ok) if ok else None,
                        "stable_items": f"{stable}/{len(per_item)}"}
    print(f"{'model':6} {'split':20} {'metric':18} {'items per repeat':22} {'mean':>6} {'range':>5}  stable")
    for (label, split, metric), v in rows.items():
        print(f"{label:6} {split:20} {metric:18} {str(v['items']):22} {v['mean']:>6} {v['range']:>5}  {v['stable_items']}")
    pairs = []
    for a, b in itertools.combinations(labels, 2):
        for (label, split, metric), v in rows.items():
            if label != a or (b, split, metric) not in rows:
                continue
            w = rows[(b, split, metric)]
            if v["mean"] is None or w["mean"] is None:
                continue
            diff = round((v["mean"] - w["mean"]) * v["sign"], 2)  # positive = a better
            spread = max(v["range"], w["range"])
            pairs.append({"a": a, "b": b, "split": split, "metric": metric, "a_minus_b_better": diff,
                          "max_range": spread, "beyond_spread": abs(diff) > spread})
    print("\npairs where the mean difference exceeds both models' own range (positive = first better):")
    for p in pairs:
        if p["beyond_spread"]:
            print(f"  {p['a']} vs {p['b']}  {p['split']:20} {p['metric']:18} {p['a_minus_b_better']:+.2f} items (range {p['max_range']})")
    json.dump({"repeats": reps, "rows": [dict(model=k[0], split=k[1], metric=k[2], **v) for k, v in rows.items()],
               "pairs": pairs}, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()

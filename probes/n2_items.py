"""N2 item-level re-check (plan.md "N2 held_out audit"): for each thinking-off headline row,
each item's mean over the three repeats for G6u and for Qwen v3, the items where the two
differ, and an exact two-sided sign test on those discordant items.

N2's rule compared row means against each model's run-to-run range. At temperature 0 an
item a model gets wrong is usually wrong every run, so one repeatable item sits "beyond
the spread" however small the row's real difference is. This asks the item question
instead: do the items where the models differ lean one way more than chance would?
A row whose metric has no per-item key in the score dicts is reported as not testable.

usage: python3 probes/n2_items.py OUT.json            (N2: G6u vs Qwen v3, thinking off)
       MODE=think python3 probes/n2_items.py OUT.json A B   (N1's thinking-on repeats, e.g. g6u g6t)
"""
import json
import math
import os
import sys

os.environ.setdefault("MODE", "nothink")  # noise_summary reads MODE at import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from noise_summary import TAGS, headline, path  # noqa: E402  (guarded by __main__)

MODELS = tuple(sys.argv[2:4]) if len(sys.argv) > 3 else ("g6u", "qwenv3")
REPS = ("r1", "r2", "r3")


def sign_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    rows = []
    for tag in TAGS:
        runs = {m: [json.load(open(path(tag, m, r))) for r in REPS] for m in MODELS}
        for split in runs[MODELS[0]][0]["summary"]:
            for metric, sign in headline(split):
                per = {}
                for m, rs in runs.items():
                    for run in rs:
                        for r in run["results"]:
                            if r["split"] == split:
                                per.setdefault(r["id"], {}).setdefault(m, []).append(r["score"].get(metric))
                row = {"set": tag, "split": split, "metric": metric, "n": len(per)}
                if all(v is None for d in per.values() for vals in d.values() for v in vals):
                    rows.append({**row, "testable": False})
                    continue
                g_better, q_better, disc = 0, 0, []
                for iid, d in sorted(per.items()):
                    a, b = MODELS
                    g = sum(bool(x) for x in d[a]) / len(d[a])
                    q = sum(bool(x) for x in d[b]) / len(d[b])
                    diff = (g - q) * sign  # positive = the first model better
                    if diff:
                        disc.append({"id": iid, a: round(g, 2), b: round(q, 2)})
                        g_better += diff > 0
                        q_better += diff < 0
                rows.append({**row, "testable": True, "g6u_better": g_better, "qwen_better": q_better,
                             "sign_p": round(sign_p(g_better, q_better), 3), "discordant": disc})
    print(f"{'set':10} {'split':20} {'metric':22} {'n':>4}  {MODELS[0]}+ / {MODELS[1]}+   sign p")
    for r in rows:
        if not r["testable"]:
            print(f"{r['set']:10} {r['split']:20} {r['metric']:22} {r['n']:>4}  not testable per item")
        elif r["g6u_better"] or r["qwen_better"]:
            mark = "  <" if r["sign_p"] < 0.05 else ""
            print(f"{r['set']:10} {r['split']:20} {r['metric']:22} {r['n']:>4}  {r['g6u_better']:>4} "
                  f"{r['qwen_better']:>5}  {r['sign_p']:>6}{mark}")
    t = [r for r in rows if r["testable"]]
    print(f"\n{len(t)} testable rows; identical item by item on {sum(not r['discordant'] for r in t)}; "
          f"p < 0.05 for {MODELS[0]} on {sum(r['sign_p'] < 0.05 and r['g6u_better'] > r['qwen_better'] for r in t)}, "
          f"for {MODELS[1]} on {sum(r['sign_p'] < 0.05 and r['qwen_better'] > r['g6u_better'] for r in t)}")
    json.dump({"models": MODELS, "repeats": REPS, "rows": rows}, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()

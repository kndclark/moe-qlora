"""Item-level comparison of two runs of one eval set (plan.md, gap ledger): for each
headline row, each item's mean over side A's files and over side B's files, the items
where they differ, and the exact two-sided sign test on those discordant items (the N2
audit's bar). A side may be one run or several repeats, comma-separated.

usage: python3 probes/pair_items.py OUT.json A.json[,A2.json...] B.json[,B2.json...]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n2_items import sign_p  # noqa: E402  (guarded by __main__)
from noise_summary import headline  # noqa: E402


def main():
    out, a_files, b_files = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
    a_runs, b_runs = [json.load(open(f)) for f in a_files], [json.load(open(f)) for f in b_files]
    rows = []
    for split in a_runs[0]["summary"]:
        for metric, sign in headline(split):
            per = {}
            for side, runs in (("a", a_runs), ("b", b_runs)):
                for run in runs:
                    for r in run["results"]:
                        if r["split"] == split:
                            per.setdefault(r["id"], {}).setdefault(side, []).append(r["score"].get(metric))
            row = {"split": split, "metric": metric, "n": len(per)}
            if all(v is None for d in per.values() for vals in d.values() for v in vals):
                rows.append({**row, "testable": False})
                continue
            a_better, b_better, disc = 0, 0, []
            for iid, d in sorted(per.items()):
                a = sum(bool(x) for x in d["a"]) / len(d["a"])
                b = sum(bool(x) for x in d["b"]) / len(d["b"])
                diff = (a - b) * sign  # positive = A better
                if diff:
                    disc.append({"id": iid, "a": round(a, 2), "b": round(b, 2)})
                    a_better += diff > 0
                    b_better += diff < 0
            rows.append({**row, "testable": True, "a_better": a_better, "b_better": b_better,
                         "sign_p": round(sign_p(a_better, b_better), 3), "discordant": disc})
    for r in rows:
        if not r["testable"]:
            print(f"{r['split']:20} {r['metric']:22} {r['n']:>4}  not testable per item")
        else:
            mark = "  <" if r["sign_p"] < 0.05 else ""
            print(f"{r['split']:20} {r['metric']:22} {r['n']:>4}  A+ {r['a_better']:>3}  B+ {r['b_better']:>3}"
                  f"  p {r['sign_p']}{mark}")
    json.dump({"a": a_files, "b": b_files, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()

"""G7c2 (plan.md "G7c2"): base Lightning's pooled v1 run (thinking off) against base's laptop
run on G6's server, per headline row, plus statuses, call formats and elapsed. One run each,
so descriptive, not a verdict: G7c's question 3 asks whether rows sit within 2 items.

usage: python3 probes/g7c2_compare.py OUT.json
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g7b_compare import items  # noqa: E402  (guarded by __main__, safe to import)
from noise_summary import headline  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = os.path.join(REPO, "results", "g7c2", "research-eval-v1-lightning-nothink-pool.json")
LAPTOP = os.path.join(REPO, "results", "research-eval-v1-lightning-nothink-g6srv.json")


def counts(run):
    fmt, st = collections.Counter(), collections.Counter()
    for r in run["results"]:
        st[r["run"]["status"]] += 1
        for c in r["run"]["calls"]:
            fmt[c.get("format", "?")] += 1
    return {"statuses": dict(st), "call_formats": dict(fmt), "elapsed_s": run.get("elapsed_s")}


def main():
    pool, lap = json.load(open(POOL)), json.load(open(LAPTOP))
    rows = []
    for split in pool["summary"]:
        for metric, sign in headline(split):
            p, l_ = items(pool, split, metric), items(lap, split, metric)
            rows.append({"split": split, "metric": metric, "sign": sign, "n": pool["summary"][split]["n"],
                         "pool": p, "laptop": l_, "diff": p - l_})
    sets = {"pool": counts(pool), "laptop": counts(lap)}
    for k, s in sets.items():
        print(f"{k:7} {s['statuses']} calls {s['call_formats']} elapsed {s['elapsed_s']}")
    print(f"\n{'split':20} {'metric':22} {'n':>4} {'pool':>5} {'laptop':>6} {'diff':>5}")
    for r in rows:
        print(f"{r['split']:20} {r['metric']:22} {r['n']:>4} {r['pool']:>5} {r['laptop']:>6} {r['diff']:>+5}")
    print(f"\n{sum(abs(r['diff']) > 2 for r in rows)} of {len(rows)} rows differ by more than 2 items")
    json.dump({"sets": sets, "rows": rows}, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()

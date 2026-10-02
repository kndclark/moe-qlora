"""G7b (plan.md "G7b"): one desktop (sm_86) thinking-off run of an adapter against its three
laptop thinking-off runs (N2: the gate's r1 plus results/noise/ r2, r3). Per headline row:
the desktop's items, the laptop's items per repeat, and how far the desktop sits outside
the laptop's [min, max]; per set: call formats, unparsed turns, statuses, elapsed.

usage: python3 probes/g7b_compare.py OUT.json LABEL
       MODE=think python3 probes/g7b_compare.py OUT.json LABEL   (G7b-think: N1's five sets)
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from noise_summary import headline  # noqa: E402  (guarded by __main__, safe to import)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODE = os.environ.get("MODE", "nothink")
SUFFIX = "think-4k" if MODE == "think" else "nothink"
TAGS = (["v2", "rocky", "promqlcat", "alert", "trap3"] if MODE == "think"
        else ["v1", "v2", "rocky", "promqlcat", "general", "alert", "trap3"])


def items(run, split, metric):
    s = run["summary"][split]
    v = s.get(metric)
    return None if v is None else round(v * s["n"])


def main():
    out_path, label = sys.argv[1], sys.argv[2]
    rows, sets = [], {}
    for tag in TAGS:
        desk = json.load(open(os.path.join(REPO, "results", "g7b", f"research-eval-{tag}-lightning-{label}-{SUFFIX}-desk.json")))
        lap = [json.load(open(os.path.join(REPO, "results", f"research-eval-{tag}-lightning-{label}-{SUFFIX}.json")))]
        lap += [json.load(open(os.path.join(REPO, "results", "noise", f"research-eval-{tag}-lightning-{label}-{SUFFIX}-{r}.json")))
                for r in ("r2", "r3")]
        fmt, st = collections.Counter(), collections.Counter()
        for r in desk["results"]:
            st[r["run"]["status"]] += 1
            for c in r["run"]["calls"]:
                fmt[c.get("format", "?")] += 1
        sets[tag] = {"statuses": dict(st), "call_formats": dict(fmt),
                     "elapsed_desk_s": desk.get("elapsed_s"), "elapsed_laptop_s": [x.get("elapsed_s") for x in lap]}
        for split in desk["summary"]:
            for metric, sign in headline(split):
                d = items(desk, split, metric)
                lv = [items(x, split, metric) for x in lap]
                lo, hi = min(lv), max(lv)
                off = 0 if lo <= d <= hi else (d - hi if d > hi else d - lo)
                rows.append({"set": tag, "split": split, "metric": metric, "sign": sign, "n": desk["summary"][split]["n"],
                             "desk": d, "laptop": lv, "outside_by": off})
    for t, s in sets.items():
        print(f"{t:10} {s['statuses']} calls {s['call_formats']} elapsed desk {s['elapsed_desk_s']} laptop {s['elapsed_laptop_s']}")
    print(f"\n{'set':10} {'split':20} {'metric':22} {'desk':>5}  laptop r1-r3   outside")
    for r in rows:
        mark = "" if r["outside_by"] == 0 else f"{r['outside_by']:+d}"
        print(f"{r['set']:10} {r['split']:20} {r['metric']:22} {r['desk']:>5}  {str(r['laptop']):14} {mark}")
    outside = [r for r in rows if r["outside_by"]]
    print(f"\n{len(outside)} of {len(rows)} rows outside the laptop's range; "
          f"{sum(abs(r['outside_by']) > 2 for r in rows)} by more than 2 items")
    json.dump({"label": label, "sets": sets, "rows": rows}, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()

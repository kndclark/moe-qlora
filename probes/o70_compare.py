"""O70's correctness reading (docs/next-model-plan.md): the 70B on the laptop's one card (TAG
o70-<arm>) against the same weights pooled across both cards (l70m), both in Meta's tool format,
on the seven sets. Every row by P1 (sign p < 0.05 on discordant items), Holm beside; plus how many
items have the same transcript word for word (final answer and every call), whether each turn
up to the first difference got a prompt of the same length, and wall time a set. Help text and
web_search (always "unavailable") are fixed, so there equal lengths mean the same tool output;
promql reads live Prometheus, whose answers change between runs, so there they do not.

usage: python3 probes/o70_compare.py o70-<arm> [ref]    (ref default l70m)
Writes results/<tag>-compare.json.
"""
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from n2_items import sign_p  # noqa: E402
from s1_compare import R, TAGS, load, model_label, p1, rows_rep  # noqa: E402

TAG = sys.argv[1]
REF = sys.argv[2] if len(sys.argv) > 2 else "l70m"
me, ref = [lambda t: model_label(TAG, "nothink", t)], [lambda t: model_label(REF, "nothink", t)]


def same(a, b):
    return a["final"] == b["final"] and [(c["name"], c["args"]) for c in a["calls"]] == \
        [(c["name"], c["args"]) for c in b["calls"]]


def inputs(a, b):  # turns up to the first that differs, and how many had equal prompt lengths:
    n = eq = 0     # equal lengths mean the tool outputs fed back were the same on both sides
    for x, y in zip(a["turns"], b["turns"]):
        n, eq = n + 1, eq + (x["prompt_tokens"] == y["prompt_tokens"])
        if x["text"] != y["text"]:
            break
    return n, eq


rows = rows_rep(me, ref, TAGS)
report = {"tag": TAG, "ref": REF, "missing": [t for t in TAGS if not load(me[0](t))]}
if rows is not None:
    ab, bb = sum(r["model_better"] for r in rows), sum(r["ref_better"] for r in rows)
    report.update(rows=rows, p1=p1(rows), pooled={"model_better": ab, "ref_better": bb,
                                                  "sign_p": round(sign_p(ab, bb), 4)})
report["sets"] = {}
for t in TAGS:
    a, b = load(me[0](t)), load(ref[0](t))
    if not (a and b):
        continue
    bid = {r["id"]: r["run"] for r in b["results"]}
    n = sum(same(r["run"], bid[r["id"]]) for r in a["results"] if r["id"] in bid)
    io = [inputs(r["run"], bid[r["id"]]) for r in a["results"] if r["id"] in bid]
    report["sets"][t] = {"items": len(a["results"]), "identical": n,
                         "turns_compared": sum(x for x, _ in io), "same_prompt_len": sum(y for _, y in io),
                         "elapsed_s": a.get("elapsed_s"), "ref_elapsed_s": b.get("elapsed_s"),
                         "status": {s: sum(r["run"]["status"] == s for r in a["results"])
                                    for s in ("answered", "call_limit", "truncated", "context_exhausted", "error")}}
json.dump(report, open(os.path.join(R, f"{TAG}-compare.json"), "w"), indent=1)

print(f"== {TAG} against {REF}" + (f"; MISSING {report['missing']}" if report["missing"] else ""))
if "p1" in report:
    q = report["p1"]
    print(f"  P1 win {q['win']} loss {q['loss']} tie {q['tie']}, Holm {q['holm']} of {q['tested']};"
          f" pooled items {report['pooled']['model_better']}/{report['pooled']['ref_better']} p {report['pooled']['sign_p']}")
    for r in rows:
        if r["model_better"] + r["ref_better"]:
            print(f"    {r['set']:9s} {r['split'] + '.' + r['metric']:38s} items +{r['model_better']}/-{r['ref_better']}"
                  f" p {r['sign_p']}" + (f"  P1 {r['p1']}" if r["p1"] != "tie" else ""))
tot = sum(s["items"] for s in report["sets"].values())
same_n = sum(s["identical"] for s in report["sets"].values())
tc = sum(s["turns_compared"] for s in report["sets"].values())
tl = sum(s["same_prompt_len"] for s in report["sets"].values())
print(f"  identical transcripts {same_n}/{tot}; turns up to the first difference with the same prompt length {tl}/{tc}")
for t, s in report["sets"].items():
    print(f"    {t:9s} identical {s['identical']:3d}/{s['items']:3d}  same input {s['same_prompt_len']}/{s['turns_compared']}"
          f"  wall {s['elapsed_s']}s (ref {s['ref_elapsed_s']}s)  {s['status']}")

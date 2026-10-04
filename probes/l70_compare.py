"""L70's readings (docs/next-model-plan.md, pre-registered 2026-10-03): untrained
Llama-3.1-70B (s1 TAG l70, thinking off, 512 tokens; Llama 3.1 has no thinking mode) against
G6q in both of its modes and against Qwen3.8-27B at low effort, on the seven sets.

Every row by P1: won or lost only at sign p < 0.05 on its discordant items, each item's score
the mean over a side's runs (s1_compare.rows_rep), the Holm count beside. Comparators:
  G6q off     its three L7 runs, all seven sets (like for like: neither side thinks)
  G6q on      its three N1 runs on N1's five sets; its one run (G6's gate) on v1 and general
  Qwen3.8 low its three runs (S1, r2, r3), all seven sets
  beside      base Lightning: N1 x3 thinking on (five sets), one run thinking off (seven)

Reading 1, "clearly ahead somewhere that matters": at least one of task, rocky_task, promql,
alert a P1 win against all three comparators on the same row, and none of the four a P1 loss
against any. Flag and trap rows are reported, not counted: G6q's data adds them.

usage: python3 probes/l70_compare.py [TAG]    (default l70)
Writes results/l70-compare.json.
"""
import collections
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from n2_items import sign_p  # noqa: E402
from s1_compare import (N1, N1_REPS, N1_SETS, R, REPEATS, TAGS, load, model_label, n1_label,  # noqa: E402
                        p1, row_items, rows_rep)

TAG = sys.argv[1] if len(sys.argv) > 1 else "l70"
REASON = [("v2", "task", "hit_and_grounded"), ("rocky", "rocky_task", "hit_and_grounded"),
          ("promqlcat", "promql", "correct"), ("alert", "alert", "correct")]
RATES = [("general", "general", "correct_where_scorable"), ("general", "general", "over_trigger"),
         ("v1", "no_tool", "correct_where_scorable"), ("v1", "no_tool", "over_trigger")]
FIVE = N1_SETS["think"]
REST = [t for t in TAGS if t not in FIVE]


def side(models, mode):
    return [lambda t, x=x: model_label(x, mode, t) for x in models]


def n1_side(fmt, reps=N1_REPS):
    return [lambda t, r=r: n1_label(fmt, t, r) for r in reps]


ME = side([TAG], "nothink")
# name -> [(reference runs, sets)]: rows_rep needs every run on every set it is given.
REFS = {
    "g6q_off": [(n1_side(N1["nothink"]["g6q"]), TAGS)],
    "g6q_on": [(n1_side(N1["think"]["g6q"]), FIVE), (n1_side(N1["think"]["g6q"], ["r1"]), REST)],
    "qwen38_low": [(side(REPEATS["q38-int4-low"], "think"), TAGS)],
}
BESIDE = {
    "lightning_on": [(n1_side(N1["think"]["lightning"]), FIVE)],
    "lightning_off": [([lambda t: f"{t}-lightning-nothink"], TAGS)],
}


def compare(parts):
    rr = []
    for fbs, sets in parts:
        got = rows_rep(ME, fbs, sets)
        if got is None:
            return None
        rr += got
    ab, bb = sum(r["model_better"] for r in rr), sum(r["ref_better"] for r in rr)
    return {"rows": rr, "p1": p1(rr), "pooled": {"model_better": ab, "ref_better": bb, "sign_p": round(sign_p(ab, bb), 4)}}


def rate(fmts, t, split, metric):
    vals = [d["summary"][split].get(metric) for d in (load(f(t)) for f in fmts) if d and split in d["summary"]]
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


runs = {t: load(model_label(TAG, "nothink", t)) for t in TAGS}
have = {t: d for t, d in runs.items() if d}
status = collections.Counter(r["run"]["status"] for d in have.values() for r in d["results"])
turns = [tn for d in have.values() for r in d["results"] for tn in r["run"]["turns"]]
items = sum(len(d["results"]) for d in have.values())
report = {"tag": TAG, "missing": [t for t in TAGS if t not in have],
          "stats": {"items": items, "status": dict(status),
                    "mean_completion_tokens": round(sum(tn["completion_tokens"] for tn in turns) / items) if items else None,
                    "turns_hit_512": sum(tn["finish_reason"] == "length" for tn in turns),
                    "max_prompt_tokens": max((tn["prompt_tokens"] for tn in turns), default=None),
                    "elapsed_s": {t: d.get("elapsed_s") for t, d in have.items()}},
          "refs": {name: compare(parts) for name, parts in {**REFS, **BESIDE}.items()}}

ref = report["refs"]
if all(ref[n] for n in REFS):
    per = {}
    for t, split, metric in REASON:
        per[split] = {n: next((r["p1"] for r in ref[n]["rows"] if r["set"] == t and r["split"] == split
                               and r["metric"] == metric), None) for n in REFS}
    won = [s for s, v in per.items() if all(x == "win" for x in v.values())]
    lost = [f"{s} vs {n}" for s, v in per.items() for n, x in v.items() if x == "loss"]
    report["reading"] = {"reasoning_p1": per, "won_against_all": won, "lost": lost,
                         "clearly_ahead": bool(won) and not lost}
sides = {TAG: ME, "g6q_off_x3": n1_side(N1["nothink"]["g6q"]), "g6q_on_n1_x3": n1_side(N1["think"]["g6q"]),
         "qwen38_low_x3": side(REPEATS["q38-int4-low"], "think"), "lightning_on_n1_x3": n1_side(N1["think"]["lightning"])}
report["reasoning_items"] = {k: row_items(v) for k, v in sides.items()}
rate_sides = {TAG: ME, "g6q_off_x3": sides["g6q_off_x3"], "g6q_on_r1": n1_side(N1["think"]["g6q"], ["r1"]),
              "qwen38_low_x3": sides["qwen38_low_x3"], "lightning_off": [lambda t: f"{t}-lightning-nothink"]}
report["rates"] = {f"{s}.{m}": {k: rate(v, t, s, m) for k, v in rate_sides.items()} for t, s, m in RATES}
json.dump(report, open(os.path.join(R, f"{TAG}-compare.json"), "w"), indent=1)


def pp(p):
    return f"{p['model_better']}/{p['ref_better']} p {p['sign_p']}"


st = report["stats"]
print(f"== {TAG}, thinking off (512): {st['items']} items {st['status']}, {st['mean_completion_tokens']} tokens/item, "
      f"{st['turns_hit_512']} turns hit 512, longest prompt {st['max_prompt_tokens']}"
      + (f"; MISSING {report['missing']}" if report["missing"] else ""))
for name, x in ref.items():
    if not x:
        print(f"  vs {name:14s} missing files")
        continue
    q = x["p1"]
    print(f"  vs {name:14s} P1 win {q['win']} loss {q['loss']} tie {q['tie']}, Holm {q['holm']} of {q['tested']};"
          f" pooled items {pp(x['pooled'])}; wins {', '.join(q['wins']) or '-'}; losses {', '.join(q['losses']) or '-'}")
    for r in x["rows"]:
        if r["p1"] != "tie" or r["model_better"] + r["ref_better"] >= 4:
            print(f"    {r['set']:9s} {r['split'] + '.' + r['metric']:38s} items +{r['model_better']}/-{r['ref_better']}"
                  f" p {r['sign_p']}" + (f"  P1 {r['p1']}" if r["p1"] != "tie" else ""))
if "reading" in report:
    rd = report["reading"]
    print(f"\n== Reading 1, clearly ahead somewhere that matters: {rd['clearly_ahead']}"
          f" (won against all three: {', '.join(rd['won_against_all']) or 'none'};"
          f" reasoning losses: {', '.join(rd['lost']) or 'none'})")
    for s, v in rd["reasoning_p1"].items():
        print(f"  {s:11s} " + "  ".join(f"{n} {x}" for n, x in v.items()))
print("\n== Reasoning rows in items: one run, or mean (range) over three")
for label, rr in report["reasoning_items"].items():
    print(f"  {label:20s} " + "  ".join(
        f"{c} {x['items'][0] if len(x['items']) == 1 else str(x['mean']) + ' (' + str(min(x['items'])) + '-' + str(max(x['items'])) + ')'}"
        for c, x in rr.items()))
print("\n== Rates (no per-item scores, or over-triggering): mean over a side's runs")
for row, v in report["rates"].items():
    print(f"  {row:36s} " + "  ".join(f"{k} {x}" for k, x in v.items()))

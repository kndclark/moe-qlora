"""Q2's readings (docs/next-model-plan.md, pre-registered 2026-10-03): Qwen3.8-27B with the
adapter trained on G6q's data (s1_screen.sh TAG q38-g6q) against base Qwen3.8 and against G6q.

Every row by P1: won or lost only at sign p < 0.05 on its discordant items, the Holm count
beside. Where a side has several runs, each item is its mean over them (s1_compare.rows_rep,
n2_items.py's way); Q2 has one run a mode.

1. Did training help? Thinking on (reasoning_effort low) against base Qwen3.8's three
   low-effort runs (S1, r2, r3); thinking off against S1's one run; all seven sets. G6's
   rule under P1, per mode: win held_out, lose none of trap, trap_control, no_tool, task,
   rocky_task, alert and promql.
2. Does it reach G6q? Thinking on against G6q's three N1 runs (N1's five sets), thinking off
   against its three L7 runs (all seven). It reaches G6q if no row is lost in either mode and
   no pooled sign test favours G6q at p < 0.05: neither mode's, nor both modes' together.
3. Does it pass G6q? The same comparison: at least one row won, none lost, and the pooled
   test over both modes together favouring Q2 at p < 0.05.
The "either mode / both together" reading of the plan's one pooled test was fixed here before
any Q2 eval ran. Beside them: base Lightning's N1 runs, N1's reasoning rows in items, and how
many items ended truncated mid-think.

usage: python3 probes/q2_compare.py [TAG]    (default q38-g6q)
Writes results/q2-compare.json.
"""
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from n2_items import sign_p  # noqa: E402
from s1_compare import (MUST_NOT_LOSE, MUST_WIN, N1, N1_REPS, N1_SETS, R, REPEATS, TAGS, load,  # noqa: E402
                        model_label, n1_label, p1, row_items, rows_rep, run_stats)

TAG = sys.argv[1] if len(sys.argv) > 1 else "q38-g6q"
Q2 = {"think": TAG + "-low", "nothink": TAG}  # model_label's names for the two passes
BASE = {"think": REPEATS["q38-int4-low"], "nothink": ["q38-int4"]}


def side(models, mode):
    return [lambda t, x=x: model_label(x, mode, t) for x in models]


def n1_side(fmt):
    return [lambda t, r=r: n1_label(fmt, t, r) for r in N1_REPS]


def compare(fa, fb, sets):
    rr = rows_rep(fa, fb, sets)
    if rr is None:
        return None
    ab, bb = sum(r["model_better"] for r in rr), sum(r["ref_better"] for r in rr)
    return {"rows": rr, "p1": p1(rr), "pooled": {"model_better": ab, "ref_better": bb, "sign_p": round(sign_p(ab, bb), 4)}}


def g6_rule(rr):
    v = {}
    for r in rr:
        v.setdefault(r["split"], []).append(r["p1"])
    reasons = [f"{s}: {sorted(set(v.get(s, [])))} (needs a win)" for s in MUST_WIN if "win" not in v.get(s, [])]
    reasons += [f"{s}: loss" for s in MUST_NOT_LOSE if "loss" in v.get(s, [])]
    return {"pass": not reasons, "reasons": reasons}


def favours(p, who):
    better, other = ("ref_better", "model_better") if who == "ref" else ("model_better", "ref_better")
    return p["sign_p"] < 0.05 and p[better] > p[other]


report = {"tag": TAG, "runs": {}, "help": {}, "g6q": {}, "lightning": {}}
for mode in ("think", "nothink"):
    runs = [load(model_label(Q2[mode], mode, t)) for t in TAGS]
    report["runs"][mode] = {"missing": [t for t, d in zip(TAGS, runs) if d is None],
                            "stats": run_stats([d for d in runs if d])}
    h = compare(side([Q2[mode]], mode), side(BASE[mode], mode), TAGS)
    if h:
        h["base_runs"] = BASE[mode]
        h["g6_rule_p1"] = g6_rule(h["rows"])
    report["help"][mode] = h
    for ref in ("g6q", "lightning"):
        fmt = N1.get(mode, {}).get(ref)
        if fmt:
            report[ref][mode] = compare(side([Q2[mode]], mode), n1_side(fmt), N1_SETS[mode])

g = [report["g6q"].get(m) for m in ("think", "nothink")]
if all(g):
    ab = sum(x["pooled"]["model_better"] for x in g)
    bb = sum(x["pooled"]["ref_better"] for x in g)
    both = {"model_better": ab, "ref_better": bb, "sign_p": round(sign_p(ab, bb), 4)}
    lost = sum(x["p1"]["loss"] for x in g)
    won = sum(x["p1"]["win"] for x in g)
    report["readings"] = {
        "1_helped": {m: report["help"][m]["g6_rule_p1"] for m in ("think", "nothink") if report["help"].get(m)},
        "2_reaches_g6q": lost == 0 and not any(favours(x["pooled"], "ref") for x in g) and not favours(both, "ref"),
        "3_passes_g6q": won >= 1 and lost == 0 and favours(both, "model"),
        "g6q_both_modes_pooled": both, "g6q_rows_won": won, "g6q_rows_lost": lost}
report["reasoning_rows"] = {
    "q2": row_items(side([Q2["think"]], "think")),
    "base_qwen38_low_x3": row_items(side(BASE["think"], "think")),
    **{f"{ref}_n1": row_items(n1_side(fmt)) for ref, fmt in N1["think"].items()}}
json.dump(report, open(os.path.join(R, "q2-compare.json"), "w"), indent=1)


def pp(p):
    return f"{p['model_better']}/{p['ref_better']} p {p['sign_p']}"


for mode in ("think", "nothink"):
    st = report["runs"][mode]
    print(f"\n== {TAG}, thinking {'on (low effort, 4096)' if mode == 'think' else 'off (512)'}: "
          f"{st['stats']['items']} items, {st['stats']['truncated_in_think']} truncated in think, "
          f"{st['stats']['mean_completion_tokens']} tokens/item" + (f"; MISSING {st['missing']}" if st["missing"] else ""))
    for name, x, extra in (("base Qwen3.8 x" + str(len(BASE[mode])), report["help"].get(mode), "help"),
                           ("G6q N1 x3", report["g6q"].get(mode), ""), ("Lightning N1 x3", report["lightning"].get(mode), "")):
        if not x:
            continue
        q = x["p1"]
        print(f"  vs {name:16s} P1 win {q['win']} loss {q['loss']} tie {q['tie']}, Holm {q['holm']} of {q['tested']};"
              f" pooled items {pp(x['pooled'])}; wins {', '.join(q['wins']) or '-'}; losses {', '.join(q['losses']) or '-'}"
              + (f"; G6 rule by P1 {'PASS' if x['g6_rule_p1']['pass'] else 'FAIL: ' + '; '.join(x['g6_rule_p1']['reasons'])}"
                 if extra else ""))
        for r in x["rows"]:
            if r["p1"] != "tie" or r["model_better"] + r["ref_better"] >= 4:
                print(f"    {r['set']:9s} {r['split'] + '.' + r['metric']:38s} items +{r['model_better']}/-{r['ref_better']}"
                      f" p {r['sign_p']}" + (f"  P1 {r['p1']}" if r["p1"] != "tie" else ""))
if "readings" in report:
    rd = report["readings"]
    print(f"\n== Readings: 1 helped {json.dumps({m: x['pass'] for m, x in rd['1_helped'].items()})};"
          f" 2 reaches G6q {rd['2_reaches_g6q']}; 3 passes G6q {rd['3_passes_g6q']}"
          f" (G6q rows won {rd['g6q_rows_won']}, lost {rd['g6q_rows_lost']}, both modes pooled {pp(rd['g6q_both_modes_pooled'])})")
print("\n== Reasoning rows, thinking on, in items: one run, or mean (range) over three")
for label, rr in report["reasoning_rows"].items():
    print(f"  {label:20s} " + "  ".join(f"{c} {x['items'][0] if len(x['items']) == 1 else str(x['mean']) + ' (' + str(min(x['items'])) + '-' + str(max(x['items'])) + ')'}"
                                       for c, x in rr.items()))

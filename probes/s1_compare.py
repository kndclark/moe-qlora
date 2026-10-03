"""S1 comparison (docs/next-model-plan.md): each screened base model against base Lightning,
the G6q adapter and (thinking off) the Qwen3-8B v3 adapter, on the seven sets.

Per headline row (G6's metrics, noise_summary.headline): the row rule of g6_compare.py,
win or loss only beyond 4 items, from the summaries; and the item-level sign test of
pair_items.py on the items both runs share. Pooled per mode: the sign test over every
discordant (row, item) pair. Also per model and mode: how many items ended truncated
mid-think, and mean completion tokens per item.

P1 (plan.md, adopted 2026-10-02; S1 pre-registered the 4-item rule a day later) is reported
beside it: a row is a win or a loss only at sign p < 0.05 on its discordant items, with
the Holm count over the rows that have per-item scores (plan.md's "Holm 0 of 14").

Against N1's repeats (plan.md N1: base Lightning and G6q, three runs each, thinking on,
the five contested sets; L7: G6q thinking off, all seven): each model's pooled sign test
against each repeat, and each reference against its own repeats, which is the null.
A model run more than once (s1_screen.sh REP=N, REPEATS below) is tested against itself
the same way. Last, N1's reasoning rows in items: each model, and the mean and range of
N1's references and of each repeated model.

The pre-registered Nano 4B training rule is applied to each Nano 4B build: thinking on,
against base Lightning, it loses at most 2 of task, rocky_task, alert, promql and general
(correct_where_scorable) -> train.

usage: python3 probes/s1_compare.py [TAG ...]    (default: every model S1 screened)
Writes results/s1-compare.json.
"""
import itertools
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from n2_items import sign_p  # noqa: E402
from noise_summary import headline  # noqa: E402

R = os.path.join(os.path.dirname(here), "results")
QWEN = os.path.expanduser("~/gpu-lab/bench/results/research-eval-")
TAGS = ["v1", "v2", "rocky", "promqlcat", "general", "alert", "trap3"]
MODELS = sys.argv[1:] or ["nano4b-bf16", "nano4b-fp8", "nano9b-bf16", "q38-int4", "q38-int4-low", "nano4b-g6q",
                          "q38-int4-r2-low", "q38-int4-r3-low"]
REASONING = [("task", "hit_and_grounded"), ("rocky_task", "hit_and_grounded"), ("alert", "correct"),
             ("promql", "correct"), ("general", "correct_where_scorable")]
# N4: an adapter screened by s1_screen.sh ADAPTER=..., and the base model it was trained on.
ADAPTED = {"nano4b-g6q": "nano4b-bf16"}
MUST_WIN = ["held_out"]  # g6_compare.py's G6 rule
MUST_NOT_LOSE = ["trap", "trap_control", "no_tool", "task", "rocky_task", "alert", "promql"]
REFS = {
    "think": {"lightning": lambda t: "v1-lightning-think-4k-g6srv" if t == "v1" else f"{t}-lightning-think-4k",
              "g6q": lambda t: f"{t}-lightning-g6q-think-4k"},
    "nothink": {"lightning": lambda t: f"{t}-lightning-nothink",
                "g6q": lambda t: f"{t}-lightning-g6q-nothink",
                "qwen3-8b-v3": lambda t: QWEN + ("L-adv3-nothink" if t == "v1" else f"{t}-L-adv3-nothink")},
}
# N1's repeats: r1 is the gate's run in results/, r2 and r3 are noise_eval.sh's in results/noise/.
N1 = {"think": {"lightning": "{t}-lightning-think-4k", "g6q": "{t}-lightning-g6q-think-4k"},
      "nothink": {"g6q": "{t}-lightning-g6q-nothink"}}
N1_SETS = {"think": ["v2", "rocky", "promqlcat", "alert", "trap3"], "nothink": TAGS}
N1_REPS = ["r1", "r2", "r3"]
N1_ROWS = [("v2", "task", "hit_and_grounded"), ("rocky", "rocky_task", "hit_and_grounded"),
           ("promqlcat", "promql", "correct"), ("alert", "alert", "correct"), ("trap3", "trap3", "noticed")]
# A model's repeat runs (s1_screen.sh REP=N), the original first.
REPEATS = {"q38-int4-low": ["q38-int4-low", "q38-int4-r2-low", "q38-int4-r3-low"]}


def n1_label(fmt, t, rep):
    label = fmt.format(t=t)
    return label if rep == "r1" else os.path.join(R, "noise", f"research-eval-{label}-{rep}")


def path(label):
    return label + ".json" if os.sep in label else os.path.join(R, f"research-eval-{label}.json")


def load(label):
    p = path(label)
    return json.load(open(p)) if os.path.exists(p) else None


def model_label(model, mode, t):
    if mode == "think":
        return f"{t}-s1-{model}-think-4k"
    return None if model.endswith("-low") else f"{t}-s1-{model}-nothink"


def rows(a, b):
    """Headline rows of run a against run b: the 4-item rule, the item-level counts and P1."""
    out = []
    for split, sa in a["summary"].items():
        sb = b["summary"].get(split)
        if sb is None:
            continue
        try:
            metrics = headline(split)
        except KeyError:
            continue
        ia = {r["id"]: r["score"] for r in a["results"] if r["split"] == split}
        ib = {r["id"]: r["score"] for r in b["results"] if r["split"] == split}
        for metric, sign in metrics:
            va, vb = sa.get(metric), sb.get(metric)
            if va is None or vb is None:
                continue
            items = round((va - vb) * sa["n"] * sign)
            ab = bb = paired = 0
            for iid in ia.keys() & ib.keys():
                x, y = ia[iid].get(metric), ib[iid].get(metric)
                if x is None or y is None:
                    continue
                paired += 1
                if bool(x) == bool(y):
                    continue
                if (bool(x) - bool(y)) * sign > 0:
                    ab += 1
                else:
                    bb += 1
            p = sign_p(ab, bb)
            out.append({"split": split, "metric": metric, "n": sa["n"], "model": va, "ref": vb,
                        "items_better": items,
                        "verdict": "win" if items > 4 else "loss" if items < -4 else "tie",
                        "model_better": ab, "ref_better": bb, "sign_p": round(p, 3),
                        "paired": paired, "p1": "tie" if p >= 0.05 else "win" if ab > bb else "loss"})
    return out


def p1(rr):
    """P1 over one comparison's rows: wins and losses at p < 0.05, and how many survive Holm."""
    ps = sorted(sign_p(r["model_better"], r["ref_better"]) for r in rr if r["paired"])
    holm = next((i for i, p in enumerate(ps) if p * (len(ps) - i) >= 0.05), len(ps))
    out = {v: sum(r["p1"] == v for r in rr) for v in ("win", "loss", "tie")}
    out.update({"holm": holm, "tested": len(ps),
                "wins": [f"{r['split']}.{r['metric']}" for r in rr if r["p1"] == "win"],
                "losses": [f"{r['split']}.{r['metric']}" for r in rr if r["p1"] == "loss"]})
    return out


def pooled(fa, fb, sets):
    """Pooled sign test of run fa against run fb (set -> label) over sets; None if a file is missing."""
    ab = bb = 0
    for t in sets:
        a, b = load(fa(t)), load(fb(t))
        if a is None or b is None:
            return None
        rr = rows(a, b)
        ab += sum(r["model_better"] for r in rr)
        bb += sum(r["ref_better"] for r in rr)
    return {"model_better": ab, "ref_better": bb, "discordant": ab + bb, "sign_p": round(sign_p(ab, bb), 4)}


def rows_rep(fas, fbs, sets):
    """Rows of several runs against several (n2_items.py's way, P1's "three repeats a side"): each
    item's mean over a side's runs, and the sign test on the items whose means differ."""
    out = []
    for t in sets:
        a, b = [load(f(t)) for f in fas], [load(f(t)) for f in fbs]
        if None in a or None in b:
            return None
        for split in a[0]["summary"]:
            try:
                metrics = headline(split)
            except KeyError:
                continue
            for metric, sign in metrics:
                per = {}
                for side, runs in ((0, a), (1, b)):
                    for d in runs:
                        for r in d["results"]:
                            if r["split"] == split:
                                per.setdefault(r["id"], ([], []))[side].append(r["score"].get(metric))
                ab = bb = paired = 0
                for x, y in per.values():
                    if len(x) != len(a) or len(y) != len(b) or None in x or None in y:
                        continue
                    paired += 1
                    diff = (sum(map(bool, x)) / len(x) - sum(map(bool, y)) / len(y)) * sign
                    ab += diff > 0
                    bb += diff < 0
                p = sign_p(ab, bb)
                out.append({"set": t, "split": split, "metric": metric, "model_better": ab, "ref_better": bb,
                            "sign_p": round(p, 3), "paired": paired,
                            "p1": "tie" if p >= 0.05 else "win" if ab > bb else "loss"})
    return out


def row_items(fmts):
    """N1's reasoning rows in items (round(value x n), as noise_summary counts) over one or more runs."""
    out = {}
    for t, split, metric in N1_ROWS:
        runs = [d for d in (load(f(t)) for f in fmts) if d]
        if not runs:
            continue
        vals = [round(d["summary"][split][metric] * d["summary"][split]["n"]) for d in runs]
        out[split] = {"n": runs[0]["summary"][split]["n"], "items": vals,
                      "mean": round(sum(vals) / len(vals), 2), "range": max(vals) - min(vals)}
    return out


def run_stats(runs):
    n = trunc = toks = 0
    for d in runs:
        for r in d["results"]:
            n += 1
            trunc += r["run"]["status"] == "truncated_in_think"
            toks += sum(t.get("completion_tokens", 0) for t in r["run"]["turns"])
    return {"items": n, "truncated_in_think": trunc, "mean_completion_tokens": round(toks / n) if n else None}


if __name__ == "__main__":  # q2_compare.py imports the helpers above
    report = {}
    for model in MODELS:
        rep = report[model] = {}
        for mode, refs in REFS.items():
            runs = {t: load(model_label(model, mode, t)) for t in TAGS if model_label(model, mode, t)}
            have = {t: d for t, d in runs.items() if d}
            if not have:
                continue
            m = rep[mode] = {"missing": [t for t in runs if t not in have], "stats": run_stats(have.values())}
            for ref, fmt in refs.items():
                rr, missing = [], []
                for t, d in have.items():
                    b = load(fmt(t))
                    if b is None:
                        missing.append(t)
                        continue
                    rr += [dict(r, set=t) for r in rows(d, b)]
                ab = sum(r["model_better"] for r in rr)
                bb = sum(r["ref_better"] for r in rr)
                m[ref] = {"rows": rr, "missing_ref": missing,
                          "verdicts": {v: sum(r["verdict"] == v for r in rr) for v in ("win", "loss", "tie")},
                          "pooled": {"model_better": ab, "ref_better": bb, "sign_p": round(sign_p(ab, bb), 4)},
                          "p1": p1(rr)}
            m["n1"] = {ref: {rn: pooled(lambda t: model_label(model, mode, t),
                                        lambda t: n1_label(fmt, t, rn), N1_SETS[mode]) for rn in N1_REPS}
                       for ref, fmt in N1.get(mode, {}).items()}
        if model in ADAPTED:  # N4's readings 1 and 2 (docs/next-model-plan.md)
            base = ADAPTED[model]
            for mode in ("think", "nothink"):
                if mode not in rep:
                    continue
                m = rep[mode]
                rr = []
                for t in TAGS:
                    a, b = load(model_label(model, mode, t)), load(model_label(base, mode, t))
                    if a and b:
                        rr += [dict(r, set=t) for r in rows(a, b)]
                ab, bb = sum(r["model_better"] for r in rr), sum(r["ref_better"] for r in rr)
                g6 = {}
                for key in ("verdict", "p1"):  # the G6 rule by the 4-item rule, then by P1
                    v = {}
                    for r in rr:
                        v.setdefault(r["split"], []).append(r[key])
                    reasons = [f"{s}: {v.get(s)} (needs a win)" for s in MUST_WIN if "win" not in v.get(s, [])]
                    reasons += [f"{s}: loss" for s in MUST_NOT_LOSE if "loss" in v.get(s, [])]
                    g6[key] = {"pass": not reasons, "reasons": reasons}
                m["base_" + base] = {"rows": rr, "verdicts": {x: sum(r["verdict"] == x for r in rr) for x in ("win", "loss", "tie")},
                                     "pooled": {"model_better": ab, "ref_better": bb, "sign_p": round(sign_p(ab, bb), 4)},
                                     "g6_rule": g6["verdict"], "p1": p1(rr), "g6_rule_p1": g6["p1"]}
                for ref in ("g6q", "lightning"):
                    x = m.get(ref)
                    if x:
                        favours_ref = x["pooled"]["sign_p"] < 0.05 and x["pooled"]["ref_better"] > x["pooled"]["model_better"]
                        x["reaches"] = x["verdicts"]["loss"] == 0 and not favours_ref
                        x["reaches_p1"] = x["p1"]["loss"] == 0 and not favours_ref
        th = rep.get("think", {}).get("lightning")
        if model.startswith("nano4b") and model not in ADAPTED and th:
            lost = [f"{s}.{k}" for s, k in REASONING
                    if any(r["split"] == s and r["metric"] == k and r["verdict"] == "loss" for r in th["rows"])]
            seen = [f"{s}.{k}" for s, k in REASONING
                    if any(r["split"] == s and r["metric"] == k for r in th["rows"])]
            lost_p1 = [f"{s}.{k}" for s, k in REASONING
                       if any(r["split"] == s and r["metric"] == k and r["p1"] == "loss" for r in th["rows"])]
            rep["training_rule"] = {"reasoning_rows_seen": seen, "lost": lost,
                                    "complete": len(seen) == len(REASONING), "train": len(lost) <= 2,
                                    "lost_p1": lost_p1, "train_p1": len(lost_p1) <= 2}

    # The null: each N1 reference against its own repeats; then each repeated model against itself.
    null = []
    for mode, refs in N1.items():
        for ref, fmt in refs.items():
            for x, y in itertools.combinations(N1_REPS, 2):
                p = pooled(lambda t: n1_label(fmt, t, x), lambda t: n1_label(fmt, t, y), N1_SETS[mode])
                null.append({"ref": ref, "mode": mode, "a": x, "b": y, "pooled": p})
    repeats = {}
    for group, runs in REPEATS.items():
        if group not in MODELS:
            continue
        g = repeats[group] = {"runs": runs, "self": [], "vs_n1": {}}
        for mode in ("think", "nothink"):
            if model_label(group, mode, "v1") is None:
                continue
            for x, y in itertools.combinations(runs, 2):
                for sets in (N1_SETS["think"], TAGS):  # N1's sets, as the null; then all seven
                    p = pooled(lambda t: model_label(x, mode, t), lambda t: model_label(y, mode, t), sets)
                    g["self"].append({"mode": mode, "a": x, "b": y, "sets": len(sets), "pooled": p})
            for ref, fmt in N1.get(mode, {}).items():  # every run a side, on N1's sets
                rr = rows_rep([lambda t, x=x: model_label(x, mode, t) for x in runs],
                              [lambda t, r=r: n1_label(fmt, t, r) for r in N1_REPS], N1_SETS[mode])
                if rr is not None:
                    ab, bb = sum(r["model_better"] for r in rr), sum(r["ref_better"] for r in rr)
                    g["vs_n1"].setdefault(mode, {})[ref] = {
                        "rows": rr, "p1": p1(rr),
                        "pooled": {"model_better": ab, "ref_better": bb, "sign_p": round(sign_p(ab, bb), 4)}}
    reasoning = {"n1": {ref: row_items([lambda t, r=r, f=fmt: n1_label(f, t, r) for r in N1_REPS])
                        for ref, fmt in N1["think"].items()},
                 "repeats": {g: row_items([lambda t, x=x: model_label(x, "think", t) for x in runs])
                             for g, runs in REPEATS.items() if g in MODELS},
                 "models": {x: row_items([lambda t, x=x: model_label(x, "think", t)]) for x in MODELS}}
    report["n1"] = {"sets": N1_SETS, "null": null, "repeats": repeats, "reasoning_rows": reasoning}
    json.dump(report, open(os.path.join(R, "s1-compare.json"), "w"), indent=1)


    def pp(p):
        return "missing" if p is None else f"{p['model_better']}/{p['ref_better']} p {p['sign_p']}"


    for model in MODELS:
        rep = report[model]
        for mode in ("think", "nothink"):
            if mode not in rep:
                continue
            m = rep[mode]
            st = m["stats"]
            print(f"\n== {model}, thinking {'on (4096)' if mode == 'think' else 'off (512)'}: {st['items']} items, "
                  f"{st['truncated_in_think']} truncated in think, {st['mean_completion_tokens']} tokens/item"
                  + (f"; missing sets: {', '.join(m['missing'])}" if m["missing"] else ""))
            for ref in REFS[mode]:
                if ref not in m:
                    continue
                x = m[ref]
                p = x["pooled"]
                print(f"  vs {ref:12s} rows win {x['verdicts']['win']} loss {x['verdicts']['loss']} tie {x['verdicts']['tie']};"
                      f" pooled items {model} +{p['model_better']} / {ref} +{p['ref_better']}, sign p {p['sign_p']}"
                      + (f"; ref missing: {', '.join(x['missing_ref'])}" if x["missing_ref"] else "")
                      + (f"; REACHES {ref}: {x['reaches']}" if "reaches" in x else ""))
                q = x["p1"]
                print(f"  {'':15s} P1 win {q['win']} loss {q['loss']} tie {q['tie']}, Holm {q['holm']} of {q['tested']}"
                      f"; wins {', '.join(q['wins']) or '-'}; losses {', '.join(q['losses']) or '-'}"
                      + (f"; REACHES {ref} by P1: {x['reaches_p1']}" if "reaches_p1" in x else ""))
            for ref, reps in m["n1"].items():
                print(f"  vs {ref:12s} N1 repeats, {len(N1_SETS[mode])} sets: "
                      + "; ".join(f"{rn} {pp(p)}" for rn, p in reps.items()))
            for key in [k for k in m if k.startswith("base_")]:
                x = m[key]
                p = x["pooled"]
                print(f"  vs {key[5:]:12s} rows win {x['verdicts']['win']} loss {x['verdicts']['loss']} tie {x['verdicts']['tie']};"
                      f" pooled items {model} +{p['model_better']} / base +{p['ref_better']}, sign p {p['sign_p']};"
                      f" G6 rule {'PASS' if x['g6_rule']['pass'] else 'FAIL: ' + '; '.join(x['g6_rule']['reasons'])}")
                q = x["p1"]
                print(f"  {'':15s} P1 win {q['win']} loss {q['loss']} tie {q['tie']}, Holm {q['holm']} of {q['tested']};"
                      f" G6 rule by P1 {'PASS' if x['g6_rule_p1']['pass'] else 'FAIL: ' + '; '.join(x['g6_rule_p1']['reasons'])}")
            for r in m.get("lightning", {}).get("rows", []):
                print(f"    {r['set']:9s} {r['split'] + '.' + r['metric']:38s} {r['model']:.3f} vs {r['ref']:.3f}"
                      f"  n={r['n']:3d} {r['items_better']:+4d} {r['verdict']:4s}  items +{r['model_better']}/-{r['ref_better']}"
                      f" p {r['sign_p']}" + (f"  P1 {r['p1']}" if r["p1"] != "tie" else ""))
        if "training_rule" in rep:
            tr = rep["training_rule"]
            print(f"  TRAINING RULE: lost {len(tr['lost'])} of {len(tr['reasoning_rows_seen'])} reasoning rows "
                  f"({', '.join(tr['lost']) or 'none'}) -> {'TRAIN' if tr['train'] else 'do not train'}"
                  + ("" if tr["complete"] else "  [INCOMPLETE: not every reasoning row ran]"))
            print(f"  TRAINING RULE by P1: lost {len(tr['lost_p1'])} ({', '.join(tr['lost_p1']) or 'none'})"
                  f" -> {'TRAIN' if tr['train_p1'] else 'do not train'}")

    n1 = report["n1"]
    print("\n== The null: each N1 reference against its own repeats (pooled items a/b, sign p)")
    for x in n1["null"]:
        print(f"  {x['ref']:9s} {x['mode']:7s} {x['a']} vs {x['b']}, {len(N1_SETS[x['mode']])} sets: {pp(x['pooled'])}")
    for group, g in n1["repeats"].items():
        print(f"\n== {group}: its repeat runs against each other")
        for x in g["self"]:
            print(f"  {x['mode']:7s} {x['a']} vs {x['b']}, {x['sets']} sets: {pp(x['pooled'])}")
        print(f"== {group}, all {len(g['runs'])} runs against all of N1's (each item's mean a side)")
        for mode, refs in g["vs_n1"].items():
            for ref, x in refs.items():
                q = x["p1"]
                print(f"  {mode:7s} vs {ref:9s} {len(N1_SETS[mode])} sets: P1 win {q['win']} loss {q['loss']} tie {q['tie']},"
                      f" Holm {q['holm']} of {q['tested']}; pooled items {pp(x['pooled'])}"
                      f"; wins {', '.join(q['wins']) or '-'}; losses {', '.join(q['losses']) or '-'}")
                for r in x["rows"]:
                    if r["model_better"] or r["ref_better"]:
                        print(f"    {r['set']:9s} {r['split'] + '.' + r['metric']:38s} items +{r['model_better']}/-{r['ref_better']}"
                              f" p {r['sign_p']}" + (f"  P1 {r['p1']}" if r["p1"] != "tie" else ""))
    print("\n== Reasoning rows, thinking on, in items: one run, or mean (range) over repeats")
    tables = [(f"{label} {'N1' if kind == 'n1' else 'x' + str(len(next(iter(rr.values()))['items']))}"
               if kind != "models" else label, rr)
              for kind, table in n1["reasoning_rows"].items() for label, rr in table.items() if rr]
    n = {c: x["n"] for _, rr in tables for c, x in rr.items()}
    print(f"  {'':22s}" + "".join(f"{c + ' (' + str(n.get(c, '?')) + ')':>16s}" for _, c, _ in N1_ROWS))
    for label, rr in tables:
        cells = ["-" if c not in rr else str(rr[c]["items"][0]) if len(rr[c]["items"]) == 1
                 else f"{rr[c]['mean']:.2f} ({min(rr[c]['items'])}-{max(rr[c]['items'])})" for _, c, _ in N1_ROWS]
        print(f"  {label:22s}" + "".join(f"{c:>16s}" for c in cells))

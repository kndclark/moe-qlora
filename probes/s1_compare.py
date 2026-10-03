"""S1 comparison (docs/next-model-plan.md): each screened base model against base Lightning,
the G6q adapter and (thinking off) the Qwen3-8B v3 adapter, on the seven sets.

Per headline row (G6's metrics, noise_summary.headline): the row rule of g6_compare.py,
win or loss only beyond 4 items, from the summaries; and the item-level sign test of
pair_items.py on the items both runs share. Pooled per mode: the sign test over every
discordant (row, item) pair. Also per model and mode: how many items ended truncated
mid-think, and mean completion tokens per item.

The pre-registered Nano 4B training rule is applied to each Nano 4B build: thinking on,
against base Lightning, it loses at most 2 of task, rocky_task, alert, promql and general
(correct_where_scorable) -> train.

usage: python3 probes/s1_compare.py [TAG ...]    (default: every model S1 screened)
Writes results/s1-compare.json.
"""
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
MODELS = sys.argv[1:] or ["nano4b-bf16", "nano4b-fp8", "nano9b-bf16", "q38-int4", "q38-int4-low"]
REASONING = [("task", "hit_and_grounded"), ("rocky_task", "hit_and_grounded"), ("alert", "correct"),
             ("promql", "correct"), ("general", "correct_where_scorable")]
REFS = {
    "think": {"lightning": lambda t: "v1-lightning-think-4k-g6srv" if t == "v1" else f"{t}-lightning-think-4k",
              "g6q": lambda t: f"{t}-lightning-g6q-think-4k"},
    "nothink": {"lightning": lambda t: f"{t}-lightning-nothink",
                "g6q": lambda t: f"{t}-lightning-g6q-nothink",
                "qwen3-8b-v3": lambda t: QWEN + ("L-adv3-nothink" if t == "v1" else f"{t}-L-adv3-nothink")},
}


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
    """Headline rows of run a against run b: the 4-item rule and the item-level counts."""
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
            ab = bb = 0
            for iid in ia.keys() & ib.keys():
                x, y = ia[iid].get(metric), ib[iid].get(metric)
                if x is None or y is None or bool(x) == bool(y):
                    continue
                if (bool(x) - bool(y)) * sign > 0:
                    ab += 1
                else:
                    bb += 1
            out.append({"split": split, "metric": metric, "n": sa["n"], "model": va, "ref": vb,
                        "items_better": items,
                        "verdict": "win" if items > 4 else "loss" if items < -4 else "tie",
                        "model_better": ab, "ref_better": bb, "sign_p": round(sign_p(ab, bb), 3)})
    return out


def run_stats(runs):
    n = trunc = toks = 0
    for d in runs:
        for r in d["results"]:
            n += 1
            trunc += r["run"]["status"] == "truncated_in_think"
            toks += sum(t.get("completion_tokens", 0) for t in r["run"]["turns"])
    return {"items": n, "truncated_in_think": trunc, "mean_completion_tokens": round(toks / n) if n else None}


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
                      "pooled": {"model_better": ab, "ref_better": bb, "sign_p": round(sign_p(ab, bb), 4)}}
    th = rep.get("think", {}).get("lightning")
    if model.startswith("nano4b") and th:
        lost = [f"{s}.{k}" for s, k in REASONING
                if any(r["split"] == s and r["metric"] == k and r["verdict"] == "loss" for r in th["rows"])]
        seen = [f"{s}.{k}" for s, k in REASONING
                if any(r["split"] == s and r["metric"] == k for r in th["rows"])]
        rep["training_rule"] = {"reasoning_rows_seen": seen, "lost": lost,
                                "complete": len(seen) == len(REASONING), "train": len(lost) <= 2}
json.dump(report, open(os.path.join(R, "s1-compare.json"), "w"), indent=1)

for model, rep in report.items():
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
                  + (f"; ref missing: {', '.join(x['missing_ref'])}" if x["missing_ref"] else ""))
        for r in m.get("lightning", {}).get("rows", []):
            print(f"    {r['set']:9s} {r['split'] + '.' + r['metric']:38s} {r['model']:.3f} vs {r['ref']:.3f}"
                  f"  n={r['n']:3d} {r['items_better']:+4d} {r['verdict']:4s}  items +{r['model_better']}/-{r['ref_better']}"
                  f" p {r['sign_p']}")
    if "training_rule" in rep:
        tr = rep["training_rule"]
        print(f"  TRAINING RULE: lost {len(tr['lost'])} of {len(tr['reasoning_rows_seen'])} reasoning rows "
              f"({', '.join(tr['lost']) or 'none'}) -> {'TRAIN' if tr['train'] else 'do not train'}"
              + ("" if tr["complete"] else "  [INCOMPLETE: not every reasoning row ran]"))

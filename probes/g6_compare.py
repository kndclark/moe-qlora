"""G6 comparison: the adapter against base Lightning, same mode, by G7a's comparison rule.

Headline metrics per split (plan.md G7a): hit_and_grounded on flag/task splits;
denied_heuristic on trap splits (higher is better) and trap_control splits (lower);
over_trigger (lower) and correct_where_scorable on no_tool/general; correct on alert and
promql; noticed (higher) and fabricated (lower) on trap3. A difference is a win or a loss
only when it exceeds 4 items on that split, otherwise a tie. Items = round(delta x n);
correct_where_scorable counts over n, which overstates it slightly (the scorable count is
not in the summary).

G6 pass (plan.md G6), thinking on primary: the adapter WINS held_out, and LOSES none of
trap, trap_control, no_tool, task, rocky_task, alert, promql. Thinking off is reported the
same way, descriptively. Also reported: base v1 on the G6 server (vLLM with LoRA enabled)
against G7a's base v1, as a check that serving with LoRA on leaves base unchanged.

usage: [LABEL=g6r] g6_compare.py [RESULTS_DIR]   writes RESULTS_DIR/<LABEL>-compare.json
LABEL is the adapter g6_eval.sh served (default g6). Any other label is also compared
with G6 itself, both modes.
"""
import json
import os
import sys

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
L = os.environ.get("LABEL", "g6")
TAGS = ["v1", "v2", "rocky", "promqlcat", "general", "alert", "trap3"]
KIND = {"held_out": "flag", "seen_tool": "flag", "trap": "trap", "trap_control": "trap",
        "no_tool": "no_tool", "held_out2": "flag", "two_flag": "flag", "fix_cmd": "flag",
        "task": "flag", "trap2": "trap", "trap2_control": "trap", "rocky_held_out": "flag",
        "rocky_task": "flag", "rocky_trap": "trap", "rocky_trap_control": "trap",
        "promql": "live", "general": "no_tool", "alert": "alert", "trap3": "trap3"}
MUST_WIN = ["held_out"]
MUST_NOT_LOSE = ["trap", "trap_control", "no_tool", "task", "rocky_task", "alert", "promql"]


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


def load(label):
    # A label with a directory in it is read from there (the Qwen runs live in gpu-lab).
    p = label + ".json" if os.sep in label else os.path.join(R, f"research-eval-{label}.json")
    return json.load(open(p))["summary"] if os.path.exists(p) else None


def compare(a_label, b_label):
    a, b = load(a_label), load(b_label)
    if a is None or b is None:
        return {"missing": [x for x, s in ((a_label, a), (b_label, b)) if s is None]}
    rows = {}
    for split in a:
        if split not in b or split not in KIND:
            continue
        n = a[split]["n"]
        for metric, sign in headline(split):
            va, vb = a[split].get(metric), b[split].get(metric)
            if va is None or vb is None:
                continue
            items = round((va - vb) * n * sign)
            rows[f"{split}.{metric}"] = {"split": split, "adapter": va, "base": vb, "n": n,
                                         "items_better": items,
                                         "verdict": "win" if items > 4 else "loss" if items < -4 else "tie"}
    return rows


def mode(adapter_fmt, base_fmt):
    rows = {}
    for tag in TAGS:
        base_label = base_fmt(tag)
        rows[tag] = compare(adapter_fmt.format(tag=tag), base_label)
    return rows


think = mode("{tag}-lightning-" + L + "-think-4k",
             lambda t: "v1-lightning-think-4k-g6srv" if t == "v1" else f"{t}-lightning-think-4k")
nothink = mode("{tag}-lightning-" + L + "-nothink", lambda t: f"{t}-lightning-nothink")
# The like-for-like question itself: both adapters, same data and recipe, same eval
# settings (thinking off, 512 tokens; G7a's yardstick runs of the Qwen3-8B v3 adapter).
QWEN = os.path.expanduser("~/gpu-lab/bench/results/research-eval-")
vs_qwen = mode("{tag}-lightning-" + L + "-nothink",
               lambda t: QWEN + ("L-adv3-nothink" if t == "v1" else f"{t}-L-adv3-nothink"))


def verdicts(rows):
    out = {}
    for tag_rows in rows.values():
        for r in tag_rows.values():
            if isinstance(r, dict) and "verdict" in r:
                out.setdefault(r["split"], []).append(r["verdict"])
    return out


def g6_pass(rows):
    v = verdicts(rows)
    missing = [s for s in MUST_WIN + MUST_NOT_LOSE if s not in v]
    reasons = [f"{s}: {v[s]} (needs a win)" for s in MUST_WIN if s in v and "win" not in v[s]]
    reasons += [f"{s}: loss" for s in MUST_NOT_LOSE if s in v and "loss" in v[s]]
    return {"pass": not reasons and not missing, "reasons": reasons, "missing_splits": missing}


check = {"think": compare("v1-lightning-think-4k-g6srv", "v1-lightning-think-4k"),
         "nothink": compare("v1-lightning-nothink-g6srv", "v1-lightning-nothink")}
report = {"think": think, "nothink": nothink, "vs_qwen_v3_nothink": vs_qwen, "pass_think": g6_pass(think),
          "nothink_rule_applied": g6_pass(nothink), "server_check_v1_base": check, "label": L}
vs_g6 = {}
if L != "g6":  # same verdict rule, G6 in the "base" column
    vs_g6 = {"think": mode("{tag}-lightning-" + L + "-think-4k", lambda t: f"{t}-lightning-g6-think-4k"),
             "nothink": mode("{tag}-lightning-" + L + "-nothink", lambda t: f"{t}-lightning-g6-nothink")}
    report["vs_g6"] = vs_g6
with open(os.path.join(R, f"{L}-compare.json"), "w") as f:
    json.dump(report, f, indent=1)

for name, rows in (("THINKING ON (primary)", think), ("THINKING OFF (secondary)", nothink),
                   (f"LIGHTNING {L} vs QWEN3-8B v3 ADAPTER, thinking off ('adapter' = Lightning)", vs_qwen),
                   *[(f"{L} vs G6 ('base' = G6), thinking {m}", r) for m, r in
                     (("on", vs_g6.get("think")), ("off", vs_g6.get("nothink"))) if r],
                   ("CHECK: base v1, G6 server vs G7a, thinking on", {"v1": check["think"]}),
                   ("CHECK: base v1, G6 server vs G7a, thinking off", {"v1": check["nothink"]})):
    print(f"\n{name}")
    for tag, tag_rows in rows.items():
        if "missing" in tag_rows:
            print(f"  {tag:10s} missing: {', '.join(tag_rows['missing'])}")
            continue
        for key, r in tag_rows.items():
            print(f"  {tag:10s} {key:36s} {r['adapter']:.3f} vs {r['base']:.3f}  n={r['n']:3d}  "
                  f"{r['items_better']:+4d} items  {r['verdict']}")
print(f"\n{L} PASS (thinking on): {report['pass_think']}")
print(f"same rule, thinking off: {report['nothink_rule_applied']}")
v = verdicts(vs_qwen)
print("vs Qwen v3 adapter: " + ", ".join(f"{k} {sum(x == k for vs in v.values() for x in vs)}" for k in ("win", "loss", "tie")))

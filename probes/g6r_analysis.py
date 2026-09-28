"""G6r (plan.md "G6R"): how the adapters use the think block, and which task items move.

Thinking on, per adapter (and base) over the seven sets: turns that never close </think>
(the answer written inside the think block), items the harness marks truncated_in_think,
turns that open with "Here's a thinking process:", and turns that close at once with empty
reasoning (the pattern G6r trains; base never produces it). Then, per task split and mode,
the hit_and_grounded items each adapter gains and loses against base, and how many of the
losses were trapped.

usage: g6r_analysis.py [LABEL ...]   default: g6 g6r; writes results/g6r-analysis.json
"""
import json
import os
import sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
TAGS = ["v1", "v2", "rocky", "promqlcat", "general", "alert", "trap3"]
TASK_SPLITS = ["task", "rocky_task"]


def load(label):
    p = os.path.join(R, f"research-eval-{label}.json")
    return json.load(open(p))["results"] if os.path.exists(p) else None


def base_label(tag, think):
    if think:
        return "v1-lightning-think-4k-g6srv" if tag == "v1" else f"{tag}-lightning-think-4k"
    return f"{tag}-lightning-nothink"


def think_use(fmt):
    st = {"items": 0, "turns": 0, "never_close": 0, "trapped_items": 0, "thinking_process_opener": 0,
          "empty_reasoning": 0, "missing": []}
    for tag in TAGS:
        res = load(fmt(tag))
        if res is None:
            st["missing"].append(fmt(tag))
            continue
        for r in res:
            st["items"] += 1
            st["trapped_items"] += r["run"]["status"] == "truncated_in_think"
            for t in r["run"]["turns"]:
                if "text" not in t:
                    continue
                text = t["text"]
                st["turns"] += 1
                st["never_close"] += "</think>" not in text
                st["thinking_process_opener"] += text.lstrip().startswith("Here's a thinking process")
                st["empty_reasoning"] += "</think>" in text and not text.split("</think>")[0].strip()
    return st


def moves(adapter_fmt, think):
    out = {}
    for tag in TAGS:
        a, b = load(adapter_fmt(tag)), load(base_label(tag, think))
        if a is None or b is None:
            continue
        base = {r["id"]: r for r in b}
        for r in a:
            if r["split"] not in TASK_SPLITS:
                continue
            ha, hb = r["score"]["hit_and_grounded"], base[r["id"]]["score"]["hit_and_grounded"]
            s = out.setdefault(r["split"], {"n": 0, "adapter": 0, "base": 0, "lost": [], "gained": [],
                                            "lost_trapped": 0})
            s["n"] += 1
            s["adapter"] += bool(ha)
            s["base"] += bool(hb)
            if hb and not ha:
                s["lost"].append(r["id"])
                s["lost_trapped"] += r["run"]["status"] == "truncated_in_think"
            elif ha and not hb:
                s["gained"].append(r["id"])
    return out


labels = sys.argv[1:] or ["g6", "g6r"]
report = {"think_use": {"base": think_use(lambda t: base_label(t, True))}, "moves": {}}
for L in labels:
    report["think_use"][L] = think_use(lambda t: f"{t}-lightning-{L}-think-4k")
    report["moves"][L] = {"think": moves(lambda t: f"{t}-lightning-{L}-think-4k", True),
                          "nothink": moves(lambda t: f"{t}-lightning-{L}-nothink", False)}
with open(os.path.join(R, "g6r-analysis.json"), "w") as f:
    json.dump(report, f, indent=1)

print("thinking on, all seven sets:")
for k, s in report["think_use"].items():
    print(f"  {k:5s} items {s['items']:4d} trapped {s['trapped_items']:3d} | turns {s['turns']:4d} "
          f"never close {s['never_close']:3d}, empty reasoning {s['empty_reasoning']:3d}, "
          f"'Here's a thinking process' {s['thinking_process_opener']:3d}"
          + (f"  MISSING {s['missing']}" if s["missing"] else ""))
for L, by_mode in report["moves"].items():
    for m, splits in by_mode.items():
        for split, s in splits.items():
            print(f"  {L:4s} {m:7s} {split:10s} {s['adapter']:2d} vs base {s['base']:2d} /{s['n']}: "
                  f"lost {len(s['lost'])} ({s['lost_trapped']} trapped), gained {len(s['gained'])}")
            if s["lost"]:
                print(f"       lost: {', '.join(s['lost'])}")

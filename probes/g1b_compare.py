"""G1b (docs/next-model-plan.md): GLM-4.7-Flash against Lightning with the allowlist stated in
the bash tool's description (research_eval.py --allowlist-in-tool, s1_screen.sh G7A=).
  readings 1-3: GLM-ad against Lightning-ad by G1's rule, per mode; with repeats, all runs a
                side (each item's mean, s1_compare.rows_rep)
  reading 4:    each arm against its own G1 run (the protocol effect), and the refusal audit
  checks:       every -ad output says allowlist_in_tool, and each item's first prompt is longer
                than G1's by the same number of tokens (the note reached the prompt)
usage: python3 g1b_compare.py [--g1]   --g1 runs the audit and rule on G1's runs instead, which
must reproduce G1's recorded table (docs/next-model-plan.md, "G1 result")
"""
import collections
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from n2_items import sign_p  # noqa: E402
from s1_compare import N1, N1_REPS, N1_SETS, R, REFS, TAGS, load, model_label, n1_label, p1, pooled, rows, rows_rep  # noqa: E402

G1 = "--g1" in sys.argv
GLM, LIGHT = ("glm47f-nvfp4", None) if G1 else ("glm47f-nvfp4-ad", "lightning-ad")
REPS = ["", "-r2", "-r3"]
# G1's audit splits: the flag lookups, where every GLM item at the call limit had a refusal.
AUDIT = {"v1": "held_out", "rocky": "rocky_held_out"}


def arm(model, mode, rep=""):
    """set -> label of an arm's run; Lightning in G1 is its reference runs."""
    if model is None:
        return REFS[mode]["lightning"]
    return lambda t: model_label(model + rep, mode, t)


def side(model, mode):
    """The runs an arm has on every set, the first run first."""
    if model is None:
        return [arm(None, mode)]
    return [arm(model, mode, r) for r in REPS if all(load(arm(model, mode, r)(t)) for t in TAGS)]


def holm_losses(rr):
    """Lost rows (P1) whose p survives Holm over all tested rows; p1()'s holm counts both ways."""
    ps = sorted((r for r in rr if r["paired"]), key=lambda r: sign_p(r["model_better"], r["ref_better"]))
    return [f"{r['set']}.{r['split']}.{r['metric']}" for r in ps[:p1(rr)["holm"]] if r["p1"] == "loss"]


def compare(fas, fbs):
    """Rows over all seven sets: one run a side by s1_compare.rows, more by rows_rep."""
    if len(fas) == len(fbs) == 1:
        rr = [dict(r, set=t) for t in TAGS for r in rows(load(fas[0](t)), load(fbs[0](t)))]
    else:
        rr = rows_rep(fas, fbs, TAGS)
    ab, bb = sum(r["model_better"] for r in rr), sum(r["ref_better"] for r in rr)
    return {"rows": rr, "p1": p1(rr), "holm_losses": holm_losses(rr), "runs": [len(fas), len(fbs)],
            "verdicts": {v: sum(r["verdict"] == v for r in rr) for v in ("win", "loss", "tie")},
            "pooled": {"model_better": ab, "ref_better": bb, "sign_p": round(sign_p(ab, bb), 4)}}


def rule(c):
    """G1's reading 1 on one comparison: beats, reaches or behind."""
    p = c["pooled"]
    if c["holm_losses"]:
        return "behind"
    if p["sign_p"] < 0.05:
        return "beats" if p["model_better"] > p["ref_better"] else "behind"
    return "reaches"


def audit(fa):
    """Items with a refused call, and those that later make a call the harness executed."""
    out = {}
    for t, split in AUDIT.items():
        n = rec = limit = 0
        for r in load(fa(t))["results"]:
            c = r["run"]["calls"]
            i = next((k for k, x in enumerate(c) if x["outcome"] == "refused"), None)
            if r["split"] != split or i is None:
                continue
            n += 1
            rec += any(x["outcome"] == "executed" for x in c[i + 1:])
            limit += r["run"]["status"] == "call_limit"
        out[t] = {"split": split, "refused_items": n, "recovered": rec, "at_call_limit": limit,
                  "recovered_pct": round(100 * rec / n) if n else None}
    return out


def first_prompt_delta(fa, fb):
    """Counter of (first prompt tokens of a) - (of b) over items both ran."""
    d = collections.Counter()
    for t in TAGS:
        b = {r["id"]: r["run"]["turns"][0]["prompt_tokens"] for r in load(fb(t))["results"] if r["run"]["turns"]}
        for r in load(fa(t))["results"]:
            if r["run"]["turns"] and r["id"] in b:
                d[r["run"]["turns"][0]["prompt_tokens"] - b[r["id"]]] += 1
    return dict(d)


def pp(p):
    return "missing" if p is None else f"+{p['model_better']} / +{p['ref_better']}, p {p['sign_p']}"


if __name__ == "__main__":
    report = {"g1": G1, "arms": [GLM, LIGHT or "lightning (G1 refs)"]}
    for mode in ("think", "nothink"):
        m = report[mode] = {}
        ga, la = side(GLM, mode), side(LIGHT, mode)
        if not ga or not la:
            print(f"== {mode}: missing runs (GLM {len(ga)}, Lightning {len(la)})")
            continue
        head = f"\n== thinking {'on (4096)' if mode == 'think' else 'off (512)'}"
        c = m["first"] = compare(ga[:1], la[:1])  # readings 1 and 3, first run a side
        m["first"]["rule"] = rule(c)
        if len(ga) > 1 or len(la) > 1:  # reading 2: every run a side
            m["all"] = compare(ga, la)
            m["all"]["rule"] = rule(m["all"])
        for key in ("first", "all"):
            if key in m:
                x, q = m[key], m[key]["p1"]
                print(f"{head}, {GLM} x{x['runs'][0]} vs {LIGHT or 'lightning'} x{x['runs'][1]}: RULE {x['rule'].upper()}"
                      f"\n  rows win {x['verdicts']['win']} loss {x['verdicts']['loss']} tie {x['verdicts']['tie']};"
                      f" pooled {pp(x['pooled'])}; P1 win {q['win']} loss {q['loss']}, Holm {q['holm']} of {q['tested']},"
                      f" Holm losses {len(x['holm_losses'])}: {', '.join(x['holm_losses']) or '-'}")
                head = " "
        for r in m["first"]["rows"]:
            if r["model_better"] or r["ref_better"]:
                print(f"    {r['set']:9s} {r['split'] + '.' + r['metric']:38s} {r['model']:.3f} vs {r['ref']:.3f}"
                      f"  items +{r['model_better']}/-{r['ref_better']} p {r['sign_p']}" + (f"  P1 {r['p1']}" if r["p1"] != "tie" else ""))
        m["audit"] = {"glm": audit(ga[0]), "lightning": audit(la[0])}
        for who, a in m["audit"].items():
            print(f"  audit {who:9s} " + "; ".join(f"{t}/{x['split']}: {x['refused_items']} refused,"
                                                    f" {x['recovered_pct']}% recovered, {x['at_call_limit']} at the call limit"
                                                    for t, x in a.items()))
        if G1:
            continue
        # Reading 4: the protocol effect, each arm against its G1 run(s); descriptive.
        olds = {"glm": [arm("glm47f-nvfp4", mode)], "lightning": [arm(None, mode)]}
        news = {"glm": ga[0], "lightning": la[0]}
        m["protocol"] = {}
        for who, fbs in olds.items():
            x = m["protocol"][who] = compare([news[who]], fbs)
            x.pop("rows")
            x["first_prompt_delta"] = first_prompt_delta(news[who], fbs[0])
            x["allowlist_in_tool"] = sorted({load(news[who](t)).get("allowlist_in_tool") is True for t in TAGS})
            x["old_audit"] = audit(fbs[0])
            print(f"  protocol {who:9s} new vs G1: pooled {pp(x['pooled'])}; P1 win {x['p1']['win']} loss {x['p1']['loss']};"
                  f" first-prompt delta {x['first_prompt_delta']}; allowlist_in_tool {x['allowlist_in_tool']}")
        fmt = N1.get(mode, {}).get("lightning")
        if fmt:  # Lightning-ad against N1's three Lightning repeats, and the repeats against each other
            m["protocol"]["lightning_n1"] = {rn: pooled(la[0], lambda t, rn=rn: n1_label(fmt, t, rn), N1_SETS[mode])
                                             for rn in N1_REPS}
            print(f"  protocol lightning vs N1, {len(N1_SETS[mode])} sets: "
                  + "; ".join(f"{rn} {pp(p)}" for rn, p in m["protocol"]["lightning_n1"].items()))
    if G1:
        sys.exit()
    out = os.path.join(R, "g1b-compare.json")
    json.dump(report, open(out, "w"), indent=1)
    print(f"\nwrote {os.path.relpath(out, os.path.dirname(R))}")

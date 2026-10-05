"""Do probes/kv_levers.sh's servers give G6q's answers? (docs/laptop-memory-levers.md)

For each phase in results/kv-levers/ and each of v1, v2, alert and trap3, against G6q's three
thinking-on reference runs (noise_summary.py's r1, r2, r3):
  agree      items whose headline scores equal each reference's
  signed     the signed headline sum (noise_summary.headline: higher is better) vs the refs'
  outside    items whose signed value matches none of the three refs, with token counts
The noise band is ref-vs-ref agreement. The check can only vouch for the adapter if base
Lightning visibly disagrees with G6q, so base-vs-G6q agreement is printed last.
Run from inside the repo (v1 has 158 items there): python3 probes/kv_levers_agree.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["MODE"] = "think"
import noise_summary as ns  # noqa: E402

K = os.path.join(HERE, "..", "results", "kv-levers", "research-eval-{}-g6q-{}.json")
PHASES = ("tm-eager", "tm-graphs092", "tm-g092-f16", "tm-g092-b512", "tm-g092-none", "tm-g092-f16-none",
          "tm-g092-f16-r2", "tm-g092-f16-r3", "tm-stack", "tm-noest", "tm-off4")
SPLITS = ("v1", "v2", "alert", "trap3")


def load(p):
    return {r["id"]: r for r in json.load(open(p))["results"]}


def tup(r):
    return tuple(r["score"].get(m) for m, _ in ns.headline(r["split"]))


def val(r):  # signed headline sum per item: higher is better
    return sum(sign * float(bool(r["score"].get(m))) for m, sign in ns.headline(r["split"]))


def agree(a, b):
    return sum(tup(a[k]) == tup(b[k]) for k in a if k in b)


def toks(r):
    return [t.get("completion_tokens") for t in (r.get("run") or {}).get("turns", [])]


def main():
    for s in SPLITS:
        refs = [load(ns.path(s, "g6q", r)) for r in ("r1", "r2", "r3")]
        rr = [agree(refs[i], refs[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
        rt = [sum(val(r) for r in ref.values()) for ref in refs]
        print(f"{s}: n={len(refs[0])} ref-vs-ref agree {'/'.join(map(str, rr))} ref signed {rt}")
        for phase in PHASES:
            p = K.format(s, phase)
            if not os.path.exists(p):
                continue
            tm = load(p)
            el = json.load(open(p)).get("elapsed_s") or 0
            ag = [agree(tm, ref) for ref in refs]
            tot = sum(val(r) for r in tm.values())
            print(f"  {phase}: n={len(tm)} agree {'/'.join(map(str, ag))} signed {tot:+.0f} wall {el:.0f}s")
            for i, r in tm.items():
                rv = [val(ref[i]) for ref in refs if i in ref]
                if rv and all(val(r) != x for x in rv):
                    print(f"     outside all refs: {i} val {val(r)} refs {rv} toks {toks(r)} "
                          f"ref toks {[toks(ref[i]) for ref in refs]}")
    print("\nbase vs G6q (the check must be able to see a missing adapter):")
    for s in SPLITS:
        refs = [load(ns.path(s, "g6q", r)) for r in ("r1", "r2", "r3")]
        bases = [load(p) for p in (ns.path(s, "base", r) for r in ("r1", "r2", "r3")) if os.path.exists(p)]
        ag = [agree(b, ref) for b in bases for ref in refs]
        print(f"  {s}: {len(bases)} base run(s) x 3 refs agree {min(ag)}-{max(ag)} of {len(refs[0])}")


if __name__ == "__main__":
    main()

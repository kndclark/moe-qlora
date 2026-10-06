"""Greedy outputs of each arm against a reference arm (REF, default base-lp2): full matches; per
prompt the first differing token, the reference's top-1 minus top-2 logprob there (a near-tie
flips on noise, a wide margin does not) and the largest gap in the chosen token's logprob over
the shared prefix (numeric drift). Files without per-position top 2 get the token check only."""
import json, os, sys
os.chdir(os.path.expanduser("~/kstage-tmp"))
ref = os.environ.get("REF", "base-lp2")
b = json.load(open(f"out/{ref}.json"))["greedy"]
print(f"reference {ref}")
for n in sys.argv[1:]:
    if not os.path.exists(f"out/{n}.json"):
        print(n, "missing"); continue
    O = json.load(open(f"out/{n}.json"))["greedy"]
    rows = []
    for x, y in zip(b, O):
        d = next((i for i, (p, q) in enumerate(zip(x["ids"], y["ids"])) if p != q), len(x["ids"]))
        t2x, t2y = x.get("top2"), y.get("top2")
        r = f"{d}"
        if t2x and d < len(x["ids"]) and t2x[d]:
            r += f"(m{t2x[d][0] - t2x[d][1]:.2f})"
        if t2x and t2y:
            r += f" d{max((abs(p[0] - q[0]) for p, q in zip(t2x[:d + 1], t2y[:d + 1]) if p and q), default=0):.2f}"
        rows.append(r)
    full = sum(x["ids"] == y["ids"] for x, y in zip(b, O))
    print(f"{n:15} full {full}/{len(b)}  first-diff(ref margin) drift: {' | '.join(rows)}")

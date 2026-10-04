"""The decision rule of docs/training-next-steps.md (1.3 and "Exit criteria"), on result files.

Rows are noise_summary.py's headline rows: one split under one headline metric; "good" flips
the metrics where lower is better (controls, over_trigger, fabricated). Unscorable items
(metric None) are left out. correct_where_scorable is a summary rate; per item it is the
"correct" field, absent where unscorable. An arm's score on an item is its mean over the
repeats.
  - primary metric: each kind (KIND: flag, trap, no_tool, live, alert, trap3) scores the mean
    of its rows' good-rates; the primary metric is the mean over the kinds present;
  - CI: paired bootstrap over tool clusters (real_tool or tool, else the item), 2,000
    resamples; p two-sided from the same resamples; Holm over the arms' comparisons with base
    (the printed CIs are unadjusted; qualifying uses the Holm-adjusted p);
  - regression flag, per row against base: the item-averaged drop exceeds both
    max(2 items, 10% of the row) and the larger of the two arms' range across repeats;
  - cost, per arm on the same runs: completion tokens per scored item (summed over its
    turns, mean over the repeats), bootstrapped with the same resamples; also the share of
    turns that hit max_tokens and the eval wall minutes;
  - qualify: Holm p < 0.05 with a positive difference, and no flag. With --named/--partner
    (a candidate's two seeds, the served one named before the test opens) the partner is
    never served, and the named seed qualifies only if the partner qualifies too;
  - serve: the leader is the qualifying arm with the highest primary metric. Another
    qualifying arm is non-inferior when its paired CI against the leader has a lower bound
    above -MARGIN (default 0.02). Among non-inferior arms, the cheapest whose tokens per
    item are at most 0.8x the leader's at the 95% bootstrap bound wins; failing that, the
    served arm (--served) if non-inferior; failing that, the leader. If none qualifies, the
    base.
Every arm must have every repeat (REPS, default "r1 r2 r3", at least 2) of every set; the
script refuses otherwise, since a dropped set changes the kinds and the cost basis
(--allow-partial runs on the complete sets and says so). On dev items this is a dry run of
the rule, not a locked-test result.

usage: [MODE=nothink] [REPS="r2 r3"] python3 probes/decision_rule.py [--served ARM]
         [--named ARM --partner ARM] [--margin 0.02] [--allow-partial] ARM [ARM ...]
       e.g. python3 probes/decision_rule.py --served g6q g6q g6u
"""
import argparse
import json
import os
import random
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from noise_summary import KIND, SUFFIX, headline, path  # noqa: E402

SETS = ["v1", "v2", "rocky", "promqlcat", "general", "alert", "trap3"]
REPS = os.environ.get("REPS", "r1 r2 r3").split()
B = 2000
FIELD = {"correct_where_scorable": "correct"}  # the summary rate's per-item field

ap = argparse.ArgumentParser(usage=__doc__.split("usage: ")[1])
ap.add_argument("arms", nargs="+")
ap.add_argument("--served")
ap.add_argument("--named")
ap.add_argument("--partner")
ap.add_argument("--margin", type=float, default=0.02)
ap.add_argument("--allow-partial", action="store_true")
o = ap.parse_args()
if bool(o.named) != bool(o.partner):
    ap.error("--named and --partner go together")
if o.partner and o.partner in (o.named, o.served):
    ap.error("the partner seed is never served: it cannot be --named or --served")
served = o.served
extra = [x for x in (o.served, o.named, o.partner) if x]
ARMS = ["base"] + [a for a in dict.fromkeys(o.arms + extra) if a != "base"]
if len(REPS) < 2:
    sys.exit("refusing: need at least 2 repeats; with one, the range condition is 0")


def load(arm, sets):
    per, clus, tok = {}, {}, {}
    cost[arm] = c = {"items": 0, "tokens": 0, "turns": 0, "capped": 0, "seconds": 0.0}
    for s in sets:
        for rep in REPS:
            d = json.load(open(path(s, arm, rep)))
            c["seconds"] += d.get("elapsed_s") or 0
            for r in d["results"]:
                turns = r["run"]["turns"]
                if any(t.get("completion_tokens") is None for t in turns):
                    sys.exit(f"refusing: {path(s, arm, rep)} item {r['id']} has a turn with no "
                             "completion_tokens; a missing count would make the arm look free")
                n = sum(t["completion_tokens"] for t in turns)
                tok.setdefault((s, r["id"]), []).append(n)
                c["items"] += 1
                c["turns"] += len(turns)
                c["tokens"] += n
                c["capped"] += sum(t.get("finish_reason") == "length" for t in turns)
                for metric, sign in headline(r["split"]):
                    v = r["score"].get(FIELD.get(metric, metric))
                    if v is None:
                        continue
                    key = (s, r["split"], metric, r["id"])
                    per.setdefault(key, []).append(int(bool(v) if sign > 0 else not v))
                    clus[key] = r.get("real_tool") or r.get("tool") or r["id"]
    return per, clus, tok


sets = [s for s in SETS if all(os.path.exists(path(s, a, r)) for a in ARMS for r in REPS)]
missing = [s for s in SETS if s not in sets]
print(f"mode {SUFFIX}; arms {ARMS}; sets with {len(REPS)} repeats for every arm: {sets}; "
      f"missing: {missing}")
if not sets:
    sys.exit("refusing: no set has every repeat for every arm")
if missing and not o.allow_partial:
    sys.exit("refusing: a missing set drops its kinds and changes the cost basis "
             "(--allow-partial to run on the complete sets)")
cost = {}
data = {a: load(a, sets) for a in ARMS}
keys = sorted(set.intersection(*({k for k, v in data[a][0].items() if len(v) == len(REPS)}
                                 for a in ARMS)))
mean = {a: {k: sum(data[a][0][k]) / len(REPS) for k in keys} for a in ARMS}
rows = sorted({k[:3] for k in keys})
by_cl = {}
for k in keys:
    by_cl.setdefault(data["base"][1][k], []).append(k)
cls = sorted(by_cl)
items_cl = {c: sorted({(k[0], k[3]) for k in v}) for c, v in by_cl.items()}
tokm = {a: {i: sum(data[a][2][i]) / len(REPS) for v in items_cl.values() for i in v}
        for a in ARMS}
tpi = {a: sum(tokm[a].values()) / len(tokm[a]) for a in ARMS}
print(f"{len(keys)} item-rows, {len(rows)} rows, {len(cls)} clusters, {len(tokm['base'])} items"
      + (" (PARTIAL: --allow-partial)" if missing else ""))


def primary(arm, w):
    rs = {}
    for k, n in w.items():
        s, m = rs.get(k[:3], (0.0, 0))
        rs[k[:3]] = (s + n * mean[arm][k], m + n)
    kinds = {}
    for row, (s, m) in rs.items():
        kinds.setdefault(KIND[row[1]], []).append(s / m)
    kv = {kd: sum(v) / len(v) for kd, v in kinds.items()}
    return sum(kv.values()) / len(kv), kv


random.seed(20261004)
boot = {a: [] for a in ARMS}
btok = {a: [] for a in ARMS}
for _ in range(B):
    w, wi = {}, {}
    for c in random.choices(cls, k=len(cls)):
        for k in by_cl[c]:
            w[k] = w.get(k, 0) + 1
        for i in items_cl[c]:
            wi[i] = wi.get(i, 0) + 1
    n = sum(wi.values())
    for a in ARMS:
        boot[a].append(primary(a, w)[0])
        btok[a].append(sum(m * tokm[a][i] for i, m in wi.items()) / n)


def ci(a, b):
    d = sorted(x - y for x, y in zip(boot[a], boot[b]))
    p = min(1.0, 2 * min(sum(x <= 0 for x in d), sum(x >= 0 for x in d)) / B)
    return d[int(B * 0.025)], d[int(B * 0.975) - 1], max(p, 1 / B)


full = {k: 1 for k in keys}
score = {}
print("\n== primary metric (kind scores)")
for a in ARMS:
    score[a], kv = primary(a, full)
    print(f"  {a:6} {score[a]:.3f}  " + "  ".join(f"{kd} {x:.3f}" for kd, x in sorted(kv.items())))

flags = {}
print("\n== regression flags against base, per row")
for a in ARMS[1:]:
    flags[a] = []
    for row in rows:
        ks = [k for k in keys if k[:3] == row]
        reps = {x: [sum(data[x][0][k][i] for k in ks) for i in range(len(REPS))] for x in ("base", a)}
        drop = sum(mean["base"][k] - mean[a][k] for k in ks)
        spread = max(max(v) - min(v) for v in reps.values())
        if drop > max(2, 0.1 * len(ks)) and drop > spread:
            flags[a].append(f"{'/'.join(row)} n={len(ks)} drop {drop:.2f} "
                            f"(base {reps['base']}, {a} {reps[a]})")
    print(f"  {a}: {len(flags[a])}" + "".join(f"\n    {f}" for f in flags[a]))

print("\n== against base: paired 95% CI, bootstrap p, Holm")
res = {a: ci(a, "base") for a in ARMS[1:]}
order = sorted(ARMS[1:], key=lambda a: res[a][2])
holm, run = {}, 0.0
for i, a in enumerate(order):
    run = max(run, min(1.0, (len(order) - i) * res[a][2]))
    holm[a] = run
for a in ARMS[1:]:
    lo, hi, p = res[a]
    print(f"  {a}-base {score[a] - score['base']:+.3f} [{lo:+.3f}, {hi:+.3f}] p {p:.4f}, Holm {holm[a]:.4f}")

print("\n== cost (all repeats and sets)")
for a in ARMS:
    c = cost[a]
    print(f"  {a:6} {tpi[a]:6.0f} completion tokens per scored item ({c['tokens'] / c['items']:.0f} "
          f"over all {c['items']}), {100 * c['capped'] / max(c['turns'], 1):.1f}% of {c['turns']} "
          f"turns hit max_tokens, {c['seconds'] / 60:.1f} min")

ok = [a for a in ARMS[1:] if holm[a] < 0.05 and score[a] > score["base"] and not flags[a]]
if o.named:
    if o.named in ok and o.partner not in ok:
        print(f"\n  {o.named} does not qualify: its partner seed {o.partner} does not")
    ok = [a for a in ok if a != o.partner and (a != o.named or o.partner in ok)]
ok.sort(key=lambda a: -score[a])
print(f"\n== qualifying arms: {ok or 'none'}")
pick = ok[0] if ok else "base"
if len(ok) > 1:
    lead, ni = ok[0], []
    for a in ok[1:]:
        lo, hi, _ = ci(a, lead)
        bound = sorted(x / y for x, y in zip(btok[a], btok[lead]))[int(B * 0.975) - 1]
        print(f"  {a}-{lead}: [{lo:+.3f}, {hi:+.3f}], "
              f"{'non-inferior' if lo > -o.margin else 'not shown non-inferior'} at -{o.margin}; "
              f"tokens {tpi[a] / tpi[lead]:.2f}x the leader's, 95% bound {bound:.2f}x")
        if lo > -o.margin:
            ni.append((a, bound))
    cheap = sorted((tpi[a], a) for a, bound in ni if bound <= 0.8)
    if cheap:
        pick = cheap[0][1]
    elif served in [a for a, _ in ni]:
        pick = served
if served and served != pick:
    lo, hi, _ = ci(served, pick)
    print(f"  served {served} - {pick}: [{lo:+.3f}, {hi:+.3f}]")
print(f"serve: {pick} (dev dry run; the locked test decides)")

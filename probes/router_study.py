"""Qwen3.8 think/no-think router study (docs/kumo-tabular.md section 4, the no-download step).

Rows are noise_summary.py's headline rows: one eval item under one headline metric, keyed
(set, split, metric, id); "good" is the metric's value, flipped for metrics where lower is
better. Arms are think-low (thinking on, reasoning_effort low, 4096 tokens) and nothink
(thinking off, 512 tokens), all on the desktop config of q38-int4 (UTIL 0.9, fp8 KV).

Protocol, fixed before repeat 3 existed:
- labels from repeats 1-2: per row, mean(think good) - mean(nothink good); ties dropped;
- held out: repeat 3 of each arm, never read when labelling;
- the gate: >= 10 replicated think-win rows (think better on the label mean AND on repeat 3);
- cheap routers, leave-one-split-out: best fixed arm, set rule, logistic regression, 5-NN;
  each is scored on repeat 3 against always-think and always-nothink.

usage: python3 probes/router_study.py                 # the study
       python3 probes/router_study.py --with-correct  # sensitivity, not the gate (below)
       python3 probes/router_study.py --check         # repeat 1 only: must give 490 rows, +55

--with-correct: no item stores correct_where_scorable (it is a summary rate), so the
pre-registered rows leave out no_tool/general correctness. This adds it from each item's
"correct" field. The gate is decided without it, as fixed before repeat 3.
"""
import json
import math
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(here)
sys.path.insert(0, here)
from noise_summary import KIND, headline  # noqa: E402

SETS = ["v1", "v2", "rocky", "promqlcat", "general", "alert", "trap3"]
THINK = ["q38-int4-low-think-4k", "q38-int4-r2-low-think-4k", "q38-int4-r5-low-think-4k"]
NOTHINK = ["q38-int4-nothink", "q38-int4-r5-nothink", "q38-int4-r6-nothink"]
FIELD = {"correct_where_scorable": "correct"} if "--with-correct" in sys.argv else {}


def load(label):
    """{row key: 1/0 good} and {row key: item} for one run over all seven sets."""
    good, items = {}, {}
    for s in SETS:
        d = json.load(open(os.path.join(REPO, "results", f"research-eval-{s}-s1-{label}.json")))
        for r in d["results"]:
            for metric, sign in headline(r["split"]):
                v = r["score"].get(FIELD.get(metric, metric))
                if v is None:  # unscorable for this item (e.g. correct_where_scorable)
                    continue
                v = bool(v)
                key = (s, r["split"], metric, r["id"])
                good[key] = int(v if sign > 0 else not v)
                items[key] = r
    return good, items


def sign_p(a, b):
    """Exact two-sided sign test on a wins vs b wins."""
    n, k = a + b, min(a, b)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wins(t, n, keys):
    tw = sum(t[k] > n[k] for k in keys)
    nw = sum(n[k] > t[k] for k in keys)
    return tw, nw, len(keys) - tw - nw


def oracle(t, n, keys):
    return sum(max(t[k], n[k]) for k in keys), sum(t[k] for k in keys), sum(n[k] for k in keys)


def features(key, item):
    s, split, metric, _ = key
    kinds = sorted(set(KIND.values()))
    f = [1.0, math.log(1 + len(item.get("question", "")))]
    f += [float(s == x) for x in SETS]
    f += [float(KIND[split] == x) for x in kinds]
    f += [float(dict(headline(split))[metric] < 0)]
    return f


def standardize(rows):
    cols = list(zip(*rows))
    mu = [sum(c) / len(c) for c in cols]
    sd = [math.sqrt(sum((x - m) ** 2 for x in c) / len(c)) or 1.0 for c, m in zip(cols, mu)]
    mu[0], sd[0] = 0.0, 1.0  # keep the bias column at 1
    return mu, sd


def logreg(X, y, steps=800, lr=0.2, l2=0.01):
    w = [0.0] * len(X[0])
    for _ in range(steps):
        g = [0.0] * len(w)
        for x, t in zip(X, y):
            z = sum(a * b for a, b in zip(w, x))
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            for j, xj in enumerate(x):
                g[j] += (p - t) * xj
        w = [wj - lr * (gj / len(X) + l2 * wj * (j > 0)) for j, (wj, gj) in enumerate(zip(w, g))]
    return w


def knn(Xtr, ytr, x, k=5, default=1):
    d = sorted((sum((a - b) ** 2 for a, b in zip(xr, x)), t) for xr, t in zip(Xtr, ytr))[:k]
    votes = sum(t for _, t in d)
    return 1 if votes * 2 > len(d) else 0 if votes * 2 < len(d) else default


def check():
    t, items = load(THINK[0])
    n, _ = load(NOTHINK[0])
    keys = sorted(set(t) & set(n))
    tw, nw, tie = wins(t, n, keys)
    o, at, an = oracle(t, n, keys)
    print(f"rows {len(keys)}, distinct items {len({k[3] for k in keys})}")
    print(f"low1 vs nothink1: think wins {tw}, nothink wins {nw}, tie {tie}")
    print(f"oracle {o} vs always-think {at}, always-nothink {an}: +{o - max(at, an)}")
    ok = (len(keys), tw, nw, tie, o - max(at, an)) == (490, 87, 55, 348, 55)
    print("check", "PASS" if ok else "FAIL (expected 490 rows, 87/55/348, +55)")
    return ok


def study():
    T = [load(l)[0] for l in THINK]
    N = [load(l)[0] for l in NOTHINK]
    items = load(THINK[0])[1]
    keys = sorted(set.intersection(*(set(g) for g in T + N)))
    print(f"rows {len(keys)} present in all {len(T)}+{len(N)} runs\n")

    print("== per repeat pair (think-low rK vs nothink rK)")
    for i in range(3):
        tw, nw, tie = wins(T[i], N[i], keys)
        o, at, an = oracle(T[i], N[i], keys)
        print(f"  pair {i + 1}: think {at}, nothink {an}; think wins {tw}, nothink wins {nw}, "
              f"tie {tie}; oracle +{o - max(at, an)}")
    for name, R in (("think-low", T), ("nothink", N)):
        flips = sum(len({R[i][k] for i in range(3)}) > 1 for k in keys)
        print(f"  {name}: totals {[sum(r[k] for k in keys) for r in R]}, rows that flip across "
              f"its 3 repeats {flips}")

    # Labels from repeats 1-2 only.
    lab = {k: (T[0][k] + T[1][k]) / 2 - (N[0][k] + N[1][k]) / 2 for k in keys}
    t3, n3 = T[2], N[2]
    think_lab = [k for k in keys if lab[k] > 0]
    noth_lab = [k for k in keys if lab[k] < 0]
    rep_t = [k for k in think_lab if t3[k] > n3[k]]
    rep_n = [k for k in noth_lab if n3[k] > t3[k]]
    strict_t = [k for k in keys if all(T[i][k] > N[i][k] for i in range(3))]
    strict_n = [k for k in keys if all(N[i][k] > T[i][k] for i in range(3))]
    print("\n== labels (repeats 1-2) replayed on repeat 3")
    print(f"  think-labelled {len(think_lab)}: think still better on repeat 3 in {len(rep_t)}, "
          f"equal in {sum(t3[k] == n3[k] for k in think_lab)}, reversed in "
          f"{sum(t3[k] < n3[k] for k in think_lab)}")
    print(f"  nothink-labelled {len(noth_lab)}: nothink still better in {len(rep_n)}, equal in "
          f"{sum(t3[k] == n3[k] for k in noth_lab)}, reversed in {sum(t3[k] > n3[k] for k in noth_lab)}")
    print(f"  replicated think wins {len(rep_t)} (gate: >= 10) -> gate "
          f"{'PASSES' if len(rep_t) >= 10 else 'FAILS'}")
    print(f"  think wins in all 3 pairs: {len(strict_t)}; nothink wins in all 3 pairs: {len(strict_n)}")
    by_set = {}
    for k in rep_t:
        by_set[k[0] + "/" + k[1]] = by_set.get(k[0] + "/" + k[1], 0) + 1
    print(f"  replicated think wins by split: {dict(sorted(by_set.items(), key=lambda x: -x[1]))}")
    by_set = {}
    for k in rep_n:
        by_set[k[0] + "/" + k[1]] = by_set.get(k[0] + "/" + k[1], 0) + 1
    print(f"  replicated nothink wins by split: {dict(sorted(by_set.items(), key=lambda x: -x[1]))}")

    # The per-item label router: an upper bound on routing that needs labels for every item.
    lab_fixed_t = sum(lab[k] for k in keys) >= 0
    route = {k: (lab[k] > 0) if lab[k] != 0 else lab_fixed_t for k in keys}
    tot = sum(t3[k] if route[k] else n3[k] for k in keys)
    at, an = sum(t3[k] for k in keys), sum(n3[k] for k in keys)
    best = t3 if at >= an else n3
    a = sum((t3[k] if route[k] else n3[k]) > best[k] for k in keys)
    b = sum((t3[k] if route[k] else n3[k]) < best[k] for k in keys)
    print("\n== per-item label router on repeat 3 (needs a label per item: an upper bound, not a router)")
    print(f"  routed {tot} vs always-think {at}, always-nothink {an}: {tot - max(at, an):+d}; "
          f"discordant vs best fixed {a}-{b}, sign p {sign_p(a, b):.4f}")

    # Cheap routers, leave-one-split-out: trained on other splits' labels, scored on repeat 3.
    splits = sorted({(k[0], k[1]) for k in keys})
    F = {k: features(k, items[k]) for k in keys}
    mu, sd = standardize([F[k] for k in keys])
    Z = {k: [(x - m) / s for x, m, s in zip(F[k], mu, sd)] for k in keys}
    pol = {"best fixed (learned)": {}, "set rule": {}, "logistic regression": {}, "5-NN": {}}
    for sp in splits:
        test = [k for k in keys if (k[0], k[1]) == sp]
        train = [k for k in keys if (k[0], k[1]) != sp and lab[k] != 0]
        y = [int(lab[k] > 0) for k in train]
        fixed = int(sum(lab[k] for k in keys if (k[0], k[1]) != sp) >= 0)
        same = [lab[k] for k in train if k[0] == sp[0]]
        setr = (int(sum(same) > 0) if sum(same) != 0 else fixed) if same else fixed
        w = logreg([Z[k] for k in train], y)
        for k in test:
            pol["best fixed (learned)"][k] = fixed
            pol["set rule"][k] = setr
            z = sum(a * b for a, b in zip(w, Z[k]))
            pol["logistic regression"][k] = int(z > 0)
            pol["5-NN"][k] = knn([Z[j] for j in train], y, Z[k], default=fixed)
    print("\n== cheap routers, leave-one-split-out, scored on repeat 3")
    print(f"  always-think {at}, always-nothink {an}")
    for name, p in pol.items():
        tot = sum(t3[k] if p[k] else n3[k] for k in keys)
        a = sum((t3[k] if p[k] else n3[k]) > best[k] for k in keys)
        b = sum((t3[k] if p[k] else n3[k]) < best[k] for k in keys)
        print(f"  {name:22} {tot} ({tot - max(at, an):+d} vs best fixed; routes "
              f"{sum(1 - p[k] for k in keys)} rows to nothink; discordant {a}-{b}, sign p {sign_p(a, b):.3f})")


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if check() else 1)
    study()

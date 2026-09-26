"""Summarise the serve A/B over every run in results/ssu_sm120/serve/r*/ and apply
plan.md's CHOSEN rule for the SSU repeats (2026-09-26): 3-run means vs triton."""
import glob, json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../results/ssu_sm120/serve")
LABELS = ["triton", "fi-cta", "fi-simple"]
def J(p): return json.load(open(p))
def row(d):
    e = J(f"{d}/research-eval-v1-nothink.json")["summary"]
    c1, c16 = J(f"{d}/bench-c1.json"), J(f"{d}/bench-c16.json")
    return {"held_out": round(e["held_out"]["hit_and_grounded"] * e["held_out"]["n"]),
            "seen_tool": round(e["seen_tool"]["hit_and_grounded"] * e["seen_tool"]["n"]),
            "truncated": sum(s.get("status", {}).get("truncated", 0) for s in e.values()),
            "c1": c1["rate_p50"], "c16": c16["rate_p50"]}
data = {L: {} for L in LABELS}
for d in sorted(glob.glob(f"{R}/r*/")):
    for L in LABELS:
        if os.path.exists(f"{d}{L}/research-eval-v1-nothink.json"):
            data[L][os.path.basename(d.rstrip("/"))] = row(f"{d}{L}")
M = ["held_out", "seen_tool", "truncated", "c1", "c16"]
print(f"{'':<10}{'run':<5}" + "".join(f"{m:>11}" for m in M))
mean, rng = {}, {}
for L in LABELS:
    for r, v in data[L].items():
        print(f"{L:<10}{r:<5}" + "".join(f"{v[m]:>11.2f}" if m in ('c1', 'c16') else f"{v[m]:>11}" for m in M))
    if data[L]:
        vals = {m: [v[m] for v in data[L].values()] for m in M}
        mean[L] = {m: sum(x) / len(x) for m, x in vals.items()}
        rng[L] = {m: max(x) - min(x) for m, x in vals.items()}
        print(f"{L:<10}{'mean':<5}" + "".join(f"{mean[L][m]:>11.2f}" for m in M)
              + f"   (n={len(data[L])}, range " + " ".join(f"{m} {rng[L][m]:.1f}" for m in M) + ")")
if "triton" not in mean: raise SystemExit
t, tr = mean["triton"], rng["triton"]
print("\nCHOSEN rule vs triton (quality: >4 items on the 3-run mean; speed: > max(5%, triton range)):")
verdict = {}
for L in LABELS[1:]:
    if L not in mean: continue
    m, q, s = mean[L], {}, {}
    for k in ("held_out", "seen_tool"):
        d = m[k] - t[k]; q[k] = "win" if d > 4 else "loss" if d < -4 else "tie"
    for k in ("c1", "c16"):
        d, bar = m[k] - t[k], max(0.05 * t[k], tr[k])
        s[k] = "faster" if d > bar else "slower" if d < -bar else "tie"
    n = min(len(data[L]), len(data["triton"]))
    verdict[L] = all(v != "loss" for v in q.values()) and s["c1"] == "faster"
    print(f"  {L:<10} n={n} " + " ".join(f"{k} {m[k] - t[k]:+.1f} {v}" for k, v in q.items())
          + " | " + " ".join(f"{k} {100 * (m[k] / t[k] - 1):+.1f}% {v}" for k, v in s.items()))
pick = "fi-simple" if verdict.get("fi-simple") else "fi-cta" if verdict.get("fi-cta") else "triton"
print(f"\nadoption (fi-simple first, then fi-cta, else triton): {pick}"
      + ("" if all(len(data[L]) >= 3 for L in LABELS) else "   [PROVISIONAL: fewer than 3 runs]"))

"""Hybrid CPU + copy engine: experts/s when both run at once vs each alone.
CPU experts/s = experts_touched / s per layer; copy experts/s = GB/s / expert bytes, averaged
over the window each kt line timed ([t_emit - calls * ms / 1000, t_emit])."""
import json, statistics, sys

H, node, B, threads = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
EXP = 5_611_520


def copy_rows(f):
    rows = [json.loads(l) for l in open(f"{H}/{f}")]
    return [r for r in rows if "gbps" in r]


def kt(f):
    out = {}
    for l in open(f"{H}/{f}"):
        t, _, rest = l.partition("\t")
        if rest.startswith("{"):
            r = json.loads(rest)
            r["t"] = float(t)
            out[r["qlen"]] = r
    return out


solo = copy_rows(f"{node}-copy-solo-a.jsonl") + copy_rows(f"{node}-copy-solo-b.jsonl")
copy_solo = statistics.mean(r["gbps"] for r in solo) * 1e9 / EXP
print(f"{node} copy solo {copy_solo:.0f} experts/s "
      f"(GB/s a {statistics.mean(r['gbps'] for r in copy_rows(f'{node}-copy-solo-a.jsonl')):.2f}, "
      f"b {statistics.mean(r['gbps'] for r in copy_rows(f'{node}-copy-solo-b.jsonl')):.2f})")
print("thr qlen cpu_solo cpu_hyb copy_hyb combined best_alone gain  cpu_slow copy_slow")
for th in threads:
    s, h = kt(f"{node}-{B}-t{th}-solo.tsv"), kt(f"{node}-{B}-t{th}-hybrid.tsv")
    during = copy_rows(f"{node}-copy-during-{B}-t{th}.jsonl")
    for q in sorted(h):
        a, b = s[q], h[q]
        lo = b["t"] - b["calls"] * b["ms_per_layer"] / 1000
        win = [r["gbps"] for r in during if lo <= r["t"] <= b["t"]]
        cpu_s = a["experts_touched"] / (a["ms_per_layer"] / 1000)
        cpu_h = b["experts_touched"] / (b["ms_per_layer"] / 1000)
        cp_h = statistics.mean(win) * 1e9 / EXP
        best = max(cpu_s, copy_solo)
        print(f"{th:>3} {q:>4} {cpu_s:8.0f} {cpu_h:7.0f} {cp_h:8.0f} {cpu_h + cp_h:8.0f} {best:10.0f} "
              f"{(cpu_h + cp_h) / best:5.2f}x {cpu_h / cpu_s - 1:+7.1%} {cp_h / copy_solo - 1:+8.1%}  n={len(win)}")

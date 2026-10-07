"""Row O, lever B1: one kernel's calls in one trace_drive.py trace. Its duration spread, how much
of each call runs alongside kernels on other streams (a graph's branches share the SMs), which
kernels those are, and what runs just before it on its own stream. p28: the fused predictor
(_ahead_kernel) takes 16-22 us a call in the server against 7.6 alone; is it waiting or sharing?
usage: trace_overlap.py TRACE_GZ [--kernel _ahead_kernel] [--top 5]
(results/kv-levers/trace-p28-*/dp0_*.pt.trace.json.gz; the .gz are not committed)"""
import argparse
import bisect
import gzip
import json
import statistics
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--kernel", default="_ahead_kernel")
ap.add_argument("--top", type=int, default=5)
a = ap.parse_args()

k = sorted((e for e in json.load(gzip.open(a.trace))["traceEvents"] if e.get("cat") == "kernel" and "dur" in e),
           key=lambda e: e["ts"])
starts = [e["ts"] for e in k]
longest = max(e["dur"] for e in k)
by_stream = defaultdict(list)
for e in k:
    by_stream[e["args"].get("stream")].append(e)
ends = {s: [e["ts"] + e["dur"] for e in v] for s, v in by_stream.items()}
calls = [e for e in k if e["name"] == a.kernel]
if not calls:
    raise SystemExit(f"no {a.kernel} in {a.trace}")
d = sorted(e["dur"] for e in calls)
q = lambda p: d[int(p * (len(d) - 1))]
print(f"{a.kernel}: {len(calls)} calls, min {d[0]:.1f} p10 {q(.1):.1f} median {q(.5):.1f} p90 {q(.9):.1f} "
      f"max {d[-1]:.1f} us")
names, olap, prev, gap = Counter(), [], Counter(), []
for c in calls:
    s, t, st = c["ts"], c["ts"] + c["dur"], c["args"].get("stream")
    o = 0.0
    for e in k[bisect.bisect_left(starts, s - longest):bisect.bisect_right(starts, t)]:
        if e["args"].get("stream") != st and e["ts"] + e["dur"] > s:
            x = min(t, e["ts"] + e["dur"]) - max(s, e["ts"])
            o += x
            names[e["name"][:60]] += x
    olap.append((c["dur"], o))
    i = bisect.bisect_right(ends[st], s + 0.01) - 1  # the last call on this stream to end before it starts
    if i >= 0 and by_stream[st][i] is not c:
        p = by_stream[st][i]
        prev[p["name"][:60]] += 1
        gap.append(s - p["ts"] - p["dur"])
fast = [o for u, o in olap if u <= q(.5)]
slow = [o for u, o in olap if u > q(.5)]
print(f"time other streams' kernels run during a call: mean {statistics.mean(o for _, o in olap):.1f} us "
      f"(faster half {statistics.mean(fast):.1f}, slower half {statistics.mean(slow) if slow else 0:.1f})")
print("those kernels, us a call:")
for n, v in names.most_common(a.top):
    print(f"  {v / len(calls):6.2f}  {n}")
print("just before it on its own stream:", ", ".join(f"{n} x{m}" for n, m in prev.most_common(3)),
      f"(gap median {statistics.median(gap):.1f} us)" if gap else "")

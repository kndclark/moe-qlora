"""Row O, lever B1: profile steady decode so every node KSTAGE_AHEAD adds to a step has a
measured GPU time. Needs a server started with --profiler-config.profiler=torch (kv_levers.sh
TRACE=1 arms). For each concurrency C: start C streams of --gen tokens (ignore_eos), wait until
all are decoding, POST /start_profile, sleep --window seconds, POST /stop_profile, wait for the
streams. One trace per concurrency lands in the server's torch_profiler_dir, in this order;
trace_ahead.py reads them.
usage: trace_drive.py --base URL --model g6q --conc 1,4,16"""
import argparse
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from offload_bench import stream  # noqa: E402


def ctl(base, path):  # /start_profile and /stop_profile answer 200 with an empty body
    req = urllib.request.Request(base + path, b"{}", {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.status

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8303")
ap.add_argument("--model", default="g6q")
ap.add_argument("--conc", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4, 16])
ap.add_argument("--gen", type=int, default=1200)
ap.add_argument("--window", type=float, default=0.6, help="seconds profiled")
a = ap.parse_args()

words = "river stone lantern copper meadow signal harbor quiet engine thread winter orbit".split()
for c in a.conc:
    got, res = [0] * c, [None] * c

    def run(i):
        p = f"Request {c}-{i}. " + " ".join(words[(i * 5 + j) % len(words)] for j in range(300)) \
            + "\nWrite a long story about these words:"
        res[i] = stream(a.base, a.model, p, a.gen, {"stream_options": {"include_usage": True}})
        got[i] = 1

    ts = [threading.Thread(target=run, args=(i,)) for i in range(c)]
    for t in ts:
        t.start()
    time.sleep(3.0)  # prefill done, every stream decoding (a 300-word prompt takes well under 1 s)
    t0 = time.time()
    ctl(a.base, "/start_profile")
    time.sleep(a.window)
    ctl(a.base, "/stop_profile")
    print(f"c={c}: profiled {a.window:g} s, start+stop took {time.time() - t0:.1f} s, "
          f"streams still running at stop: {c - sum(got)}", flush=True)
    for t in ts:
        t.join()
    tps = [r.get("decode_tps") for r in res if r and r.get("decode_tps")]
    err = [r["error"] for r in res if r and r.get("error")]
    print(f"c={c}: {len(tps)} streams, median decode {sorted(tps)[len(tps) // 2] if tps else 0:.1f} tok/s"
          f"{', errors ' + str(err[:2]) if err else ''}", flush=True)

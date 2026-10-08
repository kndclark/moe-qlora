"""Row L: where the agent loads' wall goes. Runs offload_bench.py's agent mode against a server
started with --profiler-config.profiler=torch (kv_levers.sh TRACE=1 with WORK=agent), and opens
a profiler window --at each listed second after the first request runs, so one trace per window
lands in the server's torch_profiler_dir: shallow, middle and deep contexts. Each window's
start and stop, in seconds after that first request, print as a "window" line; the bench's turn
times ("t" in per_agent) share the origin to within a poll, so the contexts in flight are known.
trace_steps.py splits each window's steps by layer type.
usage: trace_agents.py --server URL --at 15,170,320 --window 3 -- <offload_bench.py args>"""
import argparse
import os
import re
import subprocess
import sys
import time
import urllib.request

here = os.path.dirname(os.path.abspath(__file__))


def ctl(base, path):  # /start_profile and /stop_profile answer 200 with an empty body
    req = urllib.request.Request(base + path, b"{}", {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.status


def running(base):
    with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
        m = re.search(r"^vllm:num_requests_running\{[^}]*\} ([0-9.e+]+)$", r.read().decode(), re.M)
    return float(m.group(1)) if m else 0.0


ap = argparse.ArgumentParser()
ap.add_argument("--server", default="http://127.0.0.1:8303")
ap.add_argument("--at", type=lambda s: [float(x) for x in s.split(",")], default=[15, 170, 320])
ap.add_argument("--window", type=float, default=3.0, help="seconds profiled")
ap.add_argument("bench", nargs=argparse.REMAINDER, help="-- then offload_bench.py's arguments")
a = ap.parse_args()
args = a.bench[1:] if a.bench[:1] == ["--"] else a.bench

p = subprocess.Popen([sys.executable, os.path.join(here, "..", "offload_bench.py"), *args])
while p.poll() is None and running(a.server) == 0:  # corpus tokenizing runs no requests
    time.sleep(0.2)
t0 = time.time()
for at in a.at:
    while p.poll() is None and time.time() - t0 < at:
        time.sleep(0.2)
    if p.poll() is not None:
        print(f"window at {at:g} s: bench already ended", flush=True)
        break
    s = time.time() - t0
    ctl(a.server, "/start_profile")
    time.sleep(a.window)
    e = time.time() - t0
    ctl(a.server, "/stop_profile")
    print(f"  window {at:g}: {s:.1f}-{e:.1f} s, stop returned at {time.time() - t0:.1f} s", flush=True)
sys.exit(p.wait())

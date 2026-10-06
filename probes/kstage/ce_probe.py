"""Row O mechanism probe: can a CUDA graph fill expert slots on the copy engine from plans its
own kernels write? ce_helper.c's thread issues the copies; the graph signals it with a captured
cuStreamWriteValue32 and waits with cuStreamWaitValue32. Expert-sized rows (four tensors,
5.35 MiB) and a 23-layer step like Lightning's, each layer a memory-bound read (decode's shape):

  layer l: wait for l's fill, read l's slots, plan l+1's fill (data-dependent rows), signal, work

Prints the step time for no fill (helper still answers, so this includes its round trip), for
K rows a layer, and the work alone; checks every copied row against its home. Copy engine hides
the fill if the step stays near max(work, copy), not work + copy (what k11 found for SM copies).
Run in the vLLM image: python3 ce_probe.py [--k 0,1,2] [--work-mb 160] [--steps 50]
A hang (k13b's desktop run sat at 0% GPU) ends in --timeout seconds with the helper's counters,
whether its stream still has work queued, and the request flags. Levers to try:
CUDA_DEVICE_MAX_CONNECTIONS=32 (fewer streams sharing a hardware queue), CE_PRIO=1, --inflight (the
first laptop hang was in g.replay with ~20 steps queued, after K=1's first replay had worked: a full
launch queue can block the main thread inside the driver, where it may hold what the helper's copy
needs; vLLM launches about one step at a time). A hang inside a helper call ends in CE_WATCH seconds (default
--timeout) with the call's name and every Python stack. Run with docker --init and timeout -k: as
PID 1 Python ignores SIGTERM, and a thread inside a CUDA call ignores everything else."""
import argparse
import ctypes
import faulthandler
import os
import signal
import subprocess
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--k", default="0,1,2,3")
ap.add_argument("--layers", type=int, default=23)
ap.add_argument("--work-mb", type=float, default=160.0, help="MB each layer reads (decode-like: ~200 us)")
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--homes", type=int, default=24)
ap.add_argument("--slots", type=int, default=8)
ap.add_argument("--timeout", type=float, default=60.0, help="seconds any one wait may take")
ap.add_argument("--inflight", type=int, default=2, help="replays queued at most (0: no limit, as the first runs)")
a = ap.parse_args()
faulthandler.dump_traceback_later(a.timeout * 10, exit=True)  # backstop if a wait below never returns
faulthandler.register(signal.SIGUSR1, all_threads=True)  # ce_helper's watchdog raises it before exiting
os.environ.setdefault("CE_WATCH", str(a.timeout))

here = os.path.dirname(os.path.abspath(__file__))
so = "/tmp/ce_helper.so"
subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-I/usr/local/cuda/include", f"{here}/ce_helper.c", "-o", so,
                "-L/usr/local/cuda/lib64/stubs", "-lcuda", "-lpthread"], check=True)
lib = ctypes.CDLL(so)
lib.ce_start.restype = ctypes.c_void_p
lib.ce_start.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 3 + [ctypes.c_void_p] * 3
lib.ce_stats.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.ce_stop.argtypes = [ctypes.c_void_p]
cu = ctypes.CDLL("libcuda.so.1")
for f in ("cuStreamWriteValue32_v2", "cuStreamWaitValue32_v2"):
    getattr(cu, f).argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
    getattr(cu, f).restype = ctypes.c_int
WAIT_EQ = 1  # CU_STREAM_WAIT_VALUE_EQ


def memop(f, addr, v, flags=0):
    r = getattr(cu, f)(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), addr, v, flags)
    assert r == 0, f"{f}: CUresult {r}"


torch.cuda.init()
dev = torch.device("cuda")
torch.zeros(1, device=dev)
ctx = ctypes.c_void_p()
assert cu.cuCtxGetCurrent(ctypes.byref(ctx)) == 0 and ctx.value
L, E, S, maxk = a.layers, a.homes, a.slots, max(int(k) for k in a.k.split(","))
MiB = 2 ** 20
tens = [int(2.379 * MiB) // 4 * 4, int(2.379 * MiB) // 4 * 4, int(0.297 * MiB) // 4 * 4, int(0.297 * MiB) // 4 * 4]
row = sum(tens)
# One set of homes and slots per layer would be 23 x (24 + 8) x 5.35 MiB; the copies do not care
# whose rows they are, so the layers share homes and each gets its own slots.
homes = [torch.randint(-2 ** 31, 2 ** 31 - 1, (E, t // 4), dtype=torch.int32).pin_memory() for t in tens]
slots = [[torch.zeros(S, t // 4, dtype=torch.int32, device=dev) for t in tens] for _ in range(L)]
w = 1 + 2 * maxk
plan_h = torch.zeros(L, w, dtype=torch.int64, pin_memory=True)
req_h = torch.zeros(L, dtype=torch.int32, pin_memory=True)
done = torch.zeros(L, dtype=torch.int32, device=dev)


class _Arr:  # a device view of pinned memory (as vllm_kstage does)
    def __init__(self, ptr, n):
        self.__cuda_array_interface__ = {"shape": (n,), "typestr": "|u1", "data": (ptr, False), "version": 3}


plan_d = torch.as_tensor(_Arr(plan_h.data_ptr(), plan_h.numel() * 8), device=dev).view(torch.int64).view(L, w)
hb = (ctypes.c_uint64 * (L * 4))(*[h.data_ptr() for _ in range(L) for h in homes])
db = (ctypes.c_uint64 * (L * 4))(*[slots[l][t].data_ptr() for l in range(L) for t in range(4)])
rb = (ctypes.c_uint64 * (L * 4))(*[t for _ in range(L) for t in tens])
ce = lib.ce_start(ctx, L, 4, maxk, ctypes.c_void_p(plan_h.data_ptr()), ctypes.c_void_p(req_h.data_ptr()),
                  ctypes.c_void_p(done.data_ptr()), hb, db, rb)
work = torch.empty(2, int(a.work_mb * 1e6) // 2, dtype=torch.bfloat16, device=dev).normal_()  # > L2: every read is DRAM
sink = torch.zeros(L, dtype=torch.float32, device=dev)
sel = torch.zeros(1, dtype=torch.int64, device=dev)
ar = torch.arange(max(maxk, 1), device=dev)
seen = torch.zeros(L, S, 4, dtype=torch.int32, device=dev)
done_ptr, req_ptr = done.data_ptr(), req_h.data_ptr()


def step(k, fill):
    sel.add_(1)
    for l in range(L):
        if fill:
            memop("cuStreamWaitValue32_v2", done_ptr + 4 * l, 1, WAIT_EQ)
            memop("cuStreamWriteValue32_v2", done_ptr + 4 * l, 0)
            for t in range(4):
                seen[l, :, t] = slots[l][t][:, 0]  # read the slots (and keep their first word)
            n = (l + 1) % L
            if k:
                src = (sel * 7 + n * 3 + ar[:k]) % E  # data-dependent rows
                plan_d[n, 1:1 + 2 * k:2] = src
                plan_d[n, 2:2 + 2 * k:2] = ar[:k] % S
            plan_d[n, :1] = k
            memop("cuStreamWriteValue32_v2", req_ptr + 4 * n, 1)
        sink[l] = work[l % 2].sum(dtype=torch.float32)


def stats():
    st = (ctypes.c_int64 * 7)()
    lib.ce_stats(ce, st)
    return list(st)


def poll(ev, tag):
    """Wait for an event without blocking in the driver: on a hang print what the helper saw and exit."""
    t0 = time.monotonic()
    while not ev.query():
        if time.monotonic() - t0 > a.timeout:
            r, n, err, q, pr, cp, wh = stats()
            print(f"HANG at {tag}: {a.timeout:g} s; helper requests {r}, rows {n}, copies {cp}, in call {wh}, err {err}, its stream "
                  f"{'idle' if q == 0 else 'has queued work' if q == 600 else q} (priority {pr}); request "
                  f"flags {req_h.tolist()}; planned rows {plan_h[:, 0].tolist()}", flush=True)
            os._exit(3)
        time.sleep(0.0002)


def wait(tag):
    """torch.cuda.synchronize, but polled."""
    ev = torch.cuda.Event()
    ev.record()
    poll(ev, tag)
    torch.cuda.synchronize()


def graph(k, fill):
    wait(f"before capture K={k}")
    time.sleep(0.05)  # let the helper answer the last request before capture: no CUDA call may run during it
    wait(f"before capture K={k}")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        step(k, fill)
    return g


def timed(g, n, tag):
    wait(tag)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    q = []
    for _ in range(n):
        if a.inflight and len(q) >= a.inflight:
            poll(q.pop(0), tag)
        g.replay()
        if a.inflight:
            q.append(torch.cuda.Event())
            q[-1].record()
    e1.record()
    wait(tag)
    return e0.elapsed_time(e1) / n


def prime(k):
    """The first layer of a step waits for a fill the previous step's last layer planned; before
    the first replay nothing planned it, so plan it from the host."""
    plan_h[0, 0] = 0
    req_h[0] = 1


res = {}
g0 = graph(0, False)
timed(g0, 3, "work")
res["work"] = timed(g0, a.steps, "work")
print(f"CUDA_DEVICE_MAX_CONNECTIONS={os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', 'unset (8)')}, "
      f"CE_PRIO={os.environ.get('CE_PRIO', '0')}", flush=True)
print(f"work alone: {res['work']:.3f} ms a step ({L} layers x {a.work_mb:g} MB read); a row is {row / MiB:.2f} MiB"
      f" in 4 copies", flush=True)
for k in [int(x) for x in a.k.split(",")]:
    g = graph(k, True)
    prime(k)
    wait(f"prime K={k}")
    g.replay()
    wait(f"first replay K={k}")
    t = timed(g, a.steps, f"timed K={k}")
    # Check: the last replay's plans (sel = s) filled rows (sel * 7 + n * 3 + i) % E into slot i of
    # layer n, read at layer n of the next step; the final step's layer-0 plan is still in flight
    # but layers 1.. were filled during the final step itself and are in place.
    wait(f"check K={k}")
    sv = int(sel.item())
    bad = 0
    for n in range(1, L):
        for i in range(k):
            r = (sv * 7 + n * 3 + i) % E
            for ti in range(4):
                bad += int(not torch.equal(slots[n][ti][i].cpu(), homes[ti][r]))
    st = stats()
    print(f"K={k}: {t:.3f} ms a step, +{t - res['work']:.3f} over work; rows checked {(L - 1) * k}, bad {bad}; "
          f"helper requests {st[0]}, rows {st[1]}, err {st[2]}", flush=True)
    res[k] = t
lib.ce_stop(ce)
print("per layer: work {:.1f} us; ".format(1e3 * res["work"] / L)
      + "; ".join(f"K={k} +{1e3 * (res[k] - res['work']) / L:.1f} us" for k in res if k != "work"))

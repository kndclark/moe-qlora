"""vLLM plugin: experts in RAM, without Marlin reading every touched expert over PCIe.

--cpu-offload-gb N --cpu-offload-params experts leaves whole MoE layers in pinned host
memory and Marlin reads them through a UVA view: each touched expert once per decode step,
and once per block of tokens routed to it in a prefill batch. Two ways round that, chosen by
KSTAGE (unset = plugin off, vLLM untouched):

  gather   K2. Each offloaded layer's expert tensors are swapped for one device staging
           buffer shared by every layer. Before the layer's Marlin call, a Triton kernel
           copies just the experts this batch routes to from host memory into it. Batches of
           KSTAGE_DMA_M tokens or more (0 = never) copy the whole layer with the copy engine
           instead (cudaMemcpy from pinned memory, ~2x the GPU's own read rate on the laptop).
  hotcold  K5. Every MoE layer's experts are reordered by how often a profile routed to
           them, then rebuilt as one tensor whose most-used rows sit in device memory and
           the rest in host memory (CUDA VMM: one address range, two kinds of page). Routing
           ids are remapped to the new order; Marlin runs unchanged, no copies per step.
           KSTAGE_COLD_GB GiB of experts go to host memory (default: what vLLM offloaded),
           spread by KSTAGE_POLICY: global (least-routed layer/expert pairs anywhere) or
           static (same count per layer). KSTAGE_PROFILE: JSON of routing counts per decoder
           layer (expert_cache_sim.py --emit-profile); without it, the order is the expert index.
           KSTAGE_DMA_M also works here: the layer is copied whole into staging first.
  cache    K6. Hotcold's split, but the device rows are a cache: H rows pinned to the
           profile's hottest experts and KSTAGE_SLOTS slots (default all = plain LRU) that
           take whatever the batch routes to, evicting the least recently used expert it
           does not route to. Every unpinned expert keeps a home row in host memory, so a
           layer is one VMM tensor of E + slots rows and Marlin sees E + slots experts. Per
           step a one-program Triton kernel picks the slots and remaps the routing ids; misses
           are copied home -> slot before Marlin runs (misses beyond the free slots are read
           in place through UVA). KSTAGE_FREEZE_M: batches of this many tokens or more
           only remap (prefill does not churn the cache). KSTAGE_STATS=SECONDS logs hits.
           KSTAGE_EVICT=lfu evicts the expert with the lowest decayed use count instead
           (half-life ~0.69 * 2**KSTAGE_SHIFT steps, default 6, at most 10). KSTAGE_COPY=N
           copies with N programs that walk the copy list, not a program per 512 words of
           every possible copy (min(slots, routings, E) of them, launched with or without misses).
           KSTAGE_COPY_LIVE=N: N programs that count the live copies first and walk only those.
           KSTAGE_DMA_M: batches of this many tokens or more that run outside CUDA graphs (prefill
           chunks) leave the cache alone and copy every non-resident expert into staging rows on
           the copy engine, KSTAGE_DMA_BUF layers ahead (default 2; each buffer is X expert rows
           of device memory, X the most any layer has at home), so Marlin reads no host memory.
  predict  A probe, no staging (or KSTAGE_PREDICT=1 on top of any mode): how well each MoE
           layer's input, put through the next two MoE layers' gates, predicts what they route
           to, overall and on the experts KSTAGE_COLD_GB would put in host memory. Decides
           whether K6 slots are worth filling a layer or two ahead. Logs every KSTAGE_STATS s.
KSTAGE_PFFIX=1 (with or without KSTAGE) fixes a race in vLLM's own --offload-backend prefetch
that gives wrong logits when the offloaded layer count is not a multiple of the step (_pf_fix).

Mount the directory and put it on PYTHONPATH; vLLM finds the plugin through the dist-info
entry point (vllm.general_plugins):
  docker run ... -v $PWD/probes/kstage:/k:ro -e PYTHONPATH=/k -e KSTAGE=gather ...
"""
import ctypes, json, os, re, sys, time

import torch

MiB = 1 << 20
_keep = []  # VMM allocations and host views the swapped tensors point into


def log(msg):
    print(f"kstage: {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- CUDA driver helpers

def _cu():
    from cuda.bindings import driver as cu
    return cu


def ok(r):
    cu = _cu()
    err, *rest = r if isinstance(r, tuple) else (r,)
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{err}")
    return rest[0] if len(rest) == 1 else (rest or None)


def is_host(t):
    """True when the tensor's bytes are host memory (pinned/UVA or VMM host pages)."""
    if t.device.type != "cuda" or t.numel() == 0:
        return t.device.type == "cpu"
    cu = _cu()
    try:
        mt = ok(cu.cuPointerGetAttribute(cu.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE, t.data_ptr()))
    except RuntimeError:
        return False
    return int(mt) == int(cu.CUmemorytype.CU_MEMORYTYPE_HOST)


def host_alias(view):
    """A CPU tensor over the pinned host bytes behind a UVA view (for copy-engine copies)."""
    cu = _cu()
    hp = int(ok(cu.cuPointerGetAttribute(cu.CUpointer_attribute.CU_POINTER_ATTRIBUTE_HOST_POINTER, view.data_ptr())))
    n = view.numel() * view.element_size()
    buf = (ctypes.c_uint8 * n).from_address(hp)
    return torch.frombuffer(buf, dtype=torch.uint8).view(view.dtype).view(view.shape)


class _Arr:  # torch.as_tensor reads this to wrap a raw device pointer
    def __init__(self, ptr, n):
        self.__cuda_array_interface__ = {"shape": (n,), "typestr": "|u1", "data": (ptr, False), "version": 3}


def _vmm():
    """(device pages prop, host pages prop, granularity) for this context's device."""
    cu = _cu()
    torch.cuda.init()
    dev = int(ok(cu.cuCtxGetDevice()))
    L = cu.CUmemLocationType

    def prop(kind):
        p = cu.CUmemAllocationProp()
        p.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        p.location.type = kind
        p.location.id = dev if kind == L.CU_MEM_LOCATION_TYPE_DEVICE else 0
        return p
    pd, ph = prop(L.CU_MEM_LOCATION_TYPE_DEVICE), prop(L.CU_MEM_LOCATION_TYPE_HOST)
    G = ok(cu.cuMemGetAllocationGranularity(pd, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
    return dev, pd, ph, int(G)


class MixedRows:
    """E rows of row_bytes in one virtual range: rows [0, n_dev) on device pages, the rest on
    host pages. The page holding the boundary is a device page. stage=(Z, handle, size) also
    maps device pages another owner made (shared by every layer) from the page holding row Z
    on; Z must be past the host pages. Rows between are never mapped."""

    def __init__(self, E, row_bytes, n_dev, stage=None):
        cu = _cu()
        self.dev, pd, ph, G = _vmm()
        up = lambda x: (x + G - 1) // G * G
        self.nbytes = E * row_bytes
        self.size = up(self.nbytes)
        self.dsize = min(up(n_dev * row_bytes), self.size)
        self.hsize = self.size - self.dsize
        maps = [(0, self.dsize, pd, None), (self.dsize, self.hsize, ph, None)]
        self.total = self.size
        if stage:
            z, h, hs = stage
            off = z * row_bytes // G * G
            assert off >= self.size and hs % G == 0, (z, row_bytes, off, self.size, hs)
            maps.append((off, hs, None, h))
            self.total = off + hs
        self.va = ok(cu.cuMemAddressReserve(self.total, G, 0, 0))
        self.handles = []  # (offset, size, handle, owned)
        for off, sz, p, h in maps:
            if sz:
                own = h is None
                if own:
                    h = ok(cu.cuMemCreate(sz, p, 0))
                ok(cu.cuMemMap(int(self.va) + off, sz, 0, h, 0))
                self.handles.append((off, sz, h, own))
        acc = cu.CUmemAccessDesc()
        acc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        acc.location.id = self.dev
        acc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        for off, sz, _, _ in self.handles:  # mapped ranges only: the gap before the stage has none
            ok(cu.cuMemSetAccess(int(self.va) + off, sz, [acc], 1))
        self.E, self.row_bytes, self.n_dev = E, row_bytes, n_dev

    def tensor(self, dtype, shape=None):
        n = self.nbytes
        if shape is not None:
            n = 1
            for d in shape:
                n *= d
            n *= torch.empty(0, dtype=dtype).element_size()
            assert n <= self.total, (shape, n, self.total)
        t = torch.as_tensor(_Arr(int(self.va), n), device="cuda").view(dtype)
        return t.view(shape if shape is not None else (self.E, -1))

    def free(self):
        cu = _cu()
        torch.cuda.synchronize()
        for off, sz, h, own in self.handles:
            ok(cu.cuMemUnmap(int(self.va) + off, sz))
            if own:
                ok(cu.cuMemRelease(h))
        ok(cu.cuMemAddressFree(self.va, self.total))
        self.handles = []


def read_rate(t, reps=5):
    """GB/s for a GPU kernel reading every byte of t (the access pattern of weights in a GEMM)."""
    v = t.reshape(-1).view(torch.int32)
    v.sum(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        v.sum(dtype=torch.int64)
    torch.cuda.synchronize()
    return reps * t.numel() * t.element_size() / (time.perf_counter() - t0) / 1e9


# ---------------------------------------------------------------- Triton kernels

import triton
import triton.language as tl


@triton.jit
def _plan_kernel(ids_ptr, n, order_ptr, E: tl.constexpr, BE: tl.constexpr, CH: tl.constexpr):
    """order[:k] = the k distinct expert ids in ids (ascending), order[k:] = -1."""
    e = tl.arange(0, BE)
    touched = tl.zeros([BE], dtype=tl.int32)
    for s in range(0, n, CH):
        o = s + tl.arange(0, CH)
        v = tl.load(ids_ptr + o, mask=o < n, other=-1).to(tl.int32)
        hit = (v[:, None] == e[None, :]).to(tl.int32)
        touched = tl.maximum(touched, tl.max(hit, axis=0))
    touched = tl.where(e < E, touched, 0)
    cs = tl.cumsum(touched, axis=0)
    nt = tl.sum(touched, axis=0)
    tl.store(order_ptr + cs - 1, e, mask=touched > 0)
    tl.store(order_ptr + nt + (e - cs), -1, mask=(touched == 0) & (e < E))


@triton.jit
def _gather_kernel(src_ptr, dst_ptr, order_ptr, row_words, BLOCK: tl.constexpr):
    """dst[e] = src[e] for e = order[program 0]; order -1 = nothing to do."""
    e = tl.load(order_ptr + tl.program_id(0))
    o = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = (o < row_words) & (e >= 0)
    base = tl.maximum(e, 0).to(tl.int64) * row_words
    x = tl.load(src_ptr + base + o, mask=m)
    tl.store(dst_ptr + base + o, x, mask=m)


BLOCK = int(os.environ.get("KSTAGE_BLOCK", "512"))  # small blocks, many warps: more reads in flight
# over PCIe. 3090: 512/16 gathers at 12.0 GB/s = the copy engine; 8192/8 managed 7.2 (gather_sweep.py)
WARPS = int(os.environ.get("KSTAGE_WARPS", "16"))
COPY_N = int(os.environ.get("KSTAGE_COPY", "0"))  # 0: _copy_rows_kernel's grid; N: _copy_rows_few
COPY_LIVE = int(os.environ.get("KSTAGE_COPY_LIVE", "0"))  # N: _copy_rows_live on N programs (overrides COPY)


def plan(ids, E):
    order = torch.empty(E, dtype=torch.int32, device=ids.device)
    _plan_kernel[(1,)](ids.contiguous(), ids.numel(), order, E=E, BE=triton.next_power_of_2(E), CH=128)
    return order


def _words(t):
    u = t.reshape(t.shape[0], -1).view(torch.uint8)
    assert u.shape[1] % 4 == 0, f"row of {u.shape[1]} B is not whole int32 words"
    return u.view(torch.int32)


def gather(src, dst, order, n_ids=None):
    """Copy the rows order names from src to dst (same shape, expert-major)."""
    s, d = _words(src), _words(dst)
    P = min(src.shape[0], n_ids if n_ids is not None else src.shape[0])
    _gather_kernel[(P, triton.cdiv(s.shape[1], BLOCK))](s, d, order, s.shape[1], BLOCK=BLOCK, num_warps=WARPS)


@triton.jit
def _k6_assign(touched, where_ptr, home_ptr, owner_ptr, stamp_ptr, clock_ptr, freq_ptr, need_ptr, miss_ptr,
               cps_ptr, cpd_ptr, stats_ptr, S, slot0, pf_ptr, E: tl.constexpr, BE: tl.constexpr,
               BS: tl.constexpr, STATS: tl.constexpr, LFU: tl.constexpr, SHIFT: tl.constexpr, AH: tl.constexpr,
               TH: tl.constexpr):
    """_cache_kernel's step from the experts the batch routes to (touched: [BE], 1 = routed).
    Returns the copies it listed: j < na (ok), source rows, destination rows."""
    e = tl.arange(0, BE)
    em = e < E
    touched = tl.where(em, touched, 0)
    wh = tl.load(where_ptr + e, mask=em, other=0)
    hm = tl.load(home_ptr + e, mask=em, other=-1)
    miss = (touched > 0) & (hm >= 0) & (wh == hm)  # pinned experts have no home (-1)
    mi = miss.to(tl.int32)
    nm = tl.sum(mi, axis=0)
    tl.store(need_ptr + e, touched, mask=em)
    tl.store(miss_ptr + tl.cumsum(mi, axis=0) - 1, e, mask=miss)
    if LFU and AH != 1:  # frozen steps (S = 0) leave the counts alone, as they leave the stamps
        f = tl.load(freq_ptr + e, mask=em, other=0)
        tl.store(freq_ptr + e, f - (f >> SHIFT) + touched * 1048576, mask=em & (S > 0))
    if AH == 2:
        pf = tl.load(pf_ptr + e, mask=em, other=0)
        useful = tl.sum((em & (touched > 0) & ~miss & (pf == 1)).to(tl.int64), axis=0)
        polluted = tl.sum((miss & (pf == 2)).to(tl.int64), axis=0)
        tl.store(pf_ptr + e, tl.where((touched > 0) | (pf == 2), 0, pf), mask=em)
    clock = tl.load(clock_ptr)
    tl.debug_barrier()
    j = tl.arange(0, BS)
    jm = j < S
    own = tl.load(owner_ptr + j, mask=jm, other=-1)
    st = tl.load(stamp_ptr + j, mask=jm, other=0)
    used = jm & (own >= 0) & (tl.load(need_ptr + tl.maximum(own, 0), mask=jm & (own >= 0), other=0) > 0)
    cand = jm & ~used
    if LFU:  # counts stay below 2**(20 + SHIFT): int32 for SHIFT <= 10
        sc = tl.load(freq_ptr + tl.maximum(own, 0), mask=jm, other=0).to(tl.int64)
    else:
        sc = st.to(tl.int64) + 2147483648
    if AH == 1 and TH > 0:  # KSTAGE_AHEAD_STALE: an ahead fill evicts only experts gone cold
        cand = cand & ((own < 0) | (sc < TH))
    ncand = tl.sum(cand.to(tl.int32), axis=0)
    key = (sc * BE + tl.maximum(own, 0)) * BS + j
    key = tl.where(cand, key, 0x3FFFFFFFFFFFFFFF)
    slot = (tl.sort(key) % BS).to(tl.int32)  # victims, least recently used (or used) first
    na = tl.minimum(nm, ncand)
    ok = j < na
    new = tl.load(miss_ptr + j, mask=ok, other=0)
    old = tl.load(owner_ptr + slot, mask=ok, other=-1)
    h_old = tl.load(home_ptr + tl.maximum(old, 0), mask=ok & (old >= 0), other=0)
    h_new = tl.load(home_ptr + new, mask=ok, other=-1)
    if AH != 0:
        pold = tl.load(pf_ptr + tl.maximum(old, 0), mask=ok & (old >= 0), other=0)
        wasted = tl.sum((ok & (old >= 0) & (pold == 1)).to(tl.int64), axis=0)
    if STATS:
        c0 = tl.load(stats_ptr)
        c1 = tl.load(stats_ptr + 1)
        c2 = tl.load(stats_ptr + 2)
    tl.debug_barrier()  # every read of the old state before any write
    tl.store(stamp_ptr + j, clock, mask=used)
    tl.store(where_ptr + old, h_old, mask=ok & (old >= 0))
    tl.store(where_ptr + new, slot0 + slot, mask=ok)
    tl.store(owner_ptr + slot, new, mask=ok)
    tl.store(stamp_ptr + slot, clock, mask=ok)
    tl.store(cps_ptr + j, tl.where(ok, h_new, -1))
    tl.store(cpd_ptr + j, slot0 + slot, mask=ok)
    if AH == 1:  # after the real step's per-expert store above (a debug_barrier apart)
        tl.store(pf_ptr + old, tl.full([BS], 2, tl.int32), mask=ok & (old >= 0))
        tl.store(pf_ptr + new, tl.full([BS], 1, tl.int32), mask=ok)
    if AH == 2:
        tl.store(pf_ptr + old, tl.zeros([BS], tl.int32), mask=ok & (old >= 0))
    tl.store(clock_ptr, clock + 1)
    if STATS:  # calls, misses, copies; one writer per layer, so plain stores
        tl.store(stats_ptr, c0 + 1)
        tl.store(stats_ptr + 1, c1 + nm)
        tl.store(stats_ptr + 2, c2 + na)
        if AH != 0:
            tl.store(stats_ptr + 5, tl.load(stats_ptr + 5) + wasted)
        if AH == 2:
            tl.store(stats_ptr + 3, tl.load(stats_ptr + 3) + useful)
            tl.store(stats_ptr + 4, tl.load(stats_ptr + 4) + polluted)
    return ok, h_new, slot0 + slot


@triton.jit
def _cache_kernel(ids_ptr, n, out_ptr, where_ptr, home_ptr, owner_ptr, stamp_ptr, clock_ptr, freq_ptr, need_ptr,
                  miss_ptr, cps_ptr, cpd_ptr, stats_ptr, S, slot0, pf_ptr,
                  E: tl.constexpr, BE: tl.constexpr, BS: tl.constexpr, CH: tl.constexpr, STATS: tl.constexpr,
                  LFU: tl.constexpr, SHIFT: tl.constexpr, AH: tl.constexpr, TH: tl.constexpr):
    """K6, one program per layer per step. Slots whose expert the batch routes to are stamped
    with the clock; each miss (ascending id) takes the slot of the least recently used expert
    the batch does not route to (ties: lower expert id, as expert_cache_sim.py) while such
    slots last, and is listed for copying (cps row -> cpd row, cps -1 = none). Misses left
    over stay in their home row. Then out = where[ids]. LFU: the victim is instead the expert
    with the lowest decayed use count; every step each expert's count loses count >> SHIFT,
    and each one the batch routes to gains 2**20 (lfu in expert_cache_sim.py).
    AH (KSTAGE_AHEAD's tally, B5): 1 = an ahead assign, 2 = a real step with ahead on. pf per
    expert: 1 = filled ahead and not yet routed to, 2 = evicted by the ahead assign just before
    this step. A real step counts useful (routed to a pf 1 expert: a miss the fill avoided) and
    polluted (missed a pf 2 one), then clears both; either kind counts wasted (evicted a pf 1
    expert). An ahead assign leaves the LFU counts alone: a prediction is not a use.
    TH (KSTAGE_AHEAD_STALE, LFU, ahead assigns only): its victims are empty slots and experts
    whose count is below TH; the misses past them (ascending id) are left to the real step."""
    e = tl.arange(0, BE)
    touched = tl.zeros([BE], dtype=tl.int32)
    for s in range(0, n, CH):
        o = s + tl.arange(0, CH)
        v = tl.load(ids_ptr + o, mask=o < n, other=-1).to(tl.int32)
        touched = tl.maximum(touched, tl.max((v[:, None] == e[None, :]).to(tl.int32), axis=0))
    _k6_assign(touched, where_ptr, home_ptr, owner_ptr, stamp_ptr, clock_ptr, freq_ptr, need_ptr, miss_ptr,
               cps_ptr, cpd_ptr, stats_ptr, S, slot0, pf_ptr, E, BE, BS, STATS, LFU, SHIFT, AH, TH)
    tl.debug_barrier()
    for s in range(0, n, CH):
        o = s + tl.arange(0, CH)
        v = tl.load(ids_ptr + o, mask=o < n, other=0)
        w = tl.load(where_ptr + v, mask=o < n, other=0)
        tl.store(out_ptr + o, w.to(out_ptr.dtype.element_ty), mask=o < n)


@triton.jit
def _copy_rows_kernel(t_ptr, cps_ptr, cpd_ptr, row_words, BLOCK: tl.constexpr):
    """t[cpd[p]] = t[cps[p]] for p = program 0; cps -1 = nothing to do."""
    r = tl.load(cps_ptr + tl.program_id(0))
    d = tl.load(cpd_ptr + tl.program_id(0))
    o = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = (o < row_words) & (r >= 0)
    x = tl.load(t_ptr + tl.maximum(r, 0).to(tl.int64) * row_words + o, mask=m)
    tl.store(t_ptr + tl.maximum(d, 0).to(tl.int64) * row_words + o, x, mask=m)


@triton.jit
def _copy_rows_few(t_ptr, cps_ptr, cpd_ptr, P, row_words, BLOCK: tl.constexpr):
    """The same copies on a fixed grid: each program walks the P entries and takes every
    num_programs-th block of each row, so an idle entry costs a load, not a program per block."""
    for p in range(P):
        r = tl.load(cps_ptr + p)
        if r >= 0:
            d = tl.load(cpd_ptr + p)
            for o0 in range(tl.program_id(0) * BLOCK, row_words, tl.num_programs(0) * BLOCK):
                o = o0 + tl.arange(0, BLOCK)
                m = o < row_words
                x = tl.load(t_ptr + r.to(tl.int64) * row_words + o, mask=m)
                tl.store(t_ptr + d.to(tl.int64) * row_words + o, x, mask=m)


@triton.jit
def _copy_rows_live(t_ptr, cps_ptr, cpd_ptr, row_words, BS: tl.constexpr, BLOCK: tl.constexpr):
    """_copy_rows_few over the live entries only. _cache_kernel lists its copies first (cps -1
    after them), so each program counts them in one load and loops over those alone. Both
    others pay per idle entry: _copy_rows_kernel a program per block of every entry (P x 1218
    programs for a w13 row, 168 us at P = 96 with nothing to copy, p19), _copy_rows_few one
    dependent load each."""
    n = tl.sum((tl.load(cps_ptr + tl.arange(0, BS)) >= 0).to(tl.int32), axis=0)
    for p in range(n):
        r = tl.load(cps_ptr + p)
        d = tl.load(cpd_ptr + p)
        for o0 in range(tl.program_id(0) * BLOCK, row_words, tl.num_programs(0) * BLOCK):
            o = o0 + tl.arange(0, BLOCK)
            m = o < row_words
            x = tl.load(t_ptr + r.to(tl.int64) * row_words + o, mask=m)
            tl.store(t_ptr + d.to(tl.int64) * row_words + o, x, mask=m)


def _rows(t):
    """t as [rows, units] of the widest of int32/int16/uint8 that divides a row."""
    u = t.reshape(t.shape[0], -1).view(torch.uint8)
    for dt, w in ((torch.int32, 4), (torch.int16, 2), (torch.uint8, 1)):
        if u.shape[1] % w == 0:
            return u.view(dt)


class _Cache:
    """K6 state for one layer: the row every expert's weights are in now, each slot's expert
    and last use, every expert's decayed use count (evict "lfu"), and the tensors whose rows
    move with the experts."""

    def __init__(self, lay, stats=None, device="cuda", evict="lru", shift=6):
        assert evict in ("lru", "lfu") and 1 <= shift <= 10, (evict, shift)
        i32 = lambda a: torch.tensor(a, dtype=torch.int32, device=device)
        self.E, self.S, self.slot0 = len(lay["where"]), lay["S"], lay["H"]
        self.where, self.home = i32(lay["where"]), i32(lay["home"])
        self.owner, self.stamp = i32(lay["owner"] or [0]), i32(lay["stamp"] or [0])
        self.lfu, self.shift, self.freq = evict == "lfu", shift, i32(lay["freq"])
        self.BE, self.BS = triton.next_power_of_2(self.E), triton.next_power_of_2(max(self.S, 16))
        self.clock = torch.zeros(1, dtype=torch.int32, device=device)
        self.need = torch.zeros(self.BE, dtype=torch.int32, device=device)
        self.miss = torch.zeros(self.BE, dtype=torch.int32, device=device)
        self.cps = torch.full((self.BS,), -1, dtype=torch.int32, device=device)
        self.cpd = torch.zeros(self.BS, dtype=torch.int32, device=device)
        self.stats = stats  # 3 int64 the GPU writes and the CPU reads (host-mapped; 6 with ahead), or None
        self.pf, self.ah = torch.zeros(self.BE, dtype=torch.int32, device=device), 0  # B5 (_cache_kernel)
        self.stale = 0  # KSTAGE_AHEAD_STALE's count threshold (_cache_kernel's TH)
        self.rows = []

    def _assign(self, ids, S, stats, ah=0):
        out = torch.empty_like(ids)
        _cache_kernel[(1,)](ids, ids.numel(), out, self.where, self.home, self.owner, self.stamp, self.clock,
                            self.freq, self.need, self.miss, self.cps, self.cpd,
                            stats if stats is not None else self.clock, S, self.slot0, self.pf, E=self.E,
                            BE=self.BE, BS=self.BS, CH=128, STATS=stats is not None, LFU=self.lfu,
                            SHIFT=self.shift, AH=ah, TH=self.stale if ah == 1 else 0, num_warps=8)
        return out

    def step(self, topk_ids, frozen=False):
        ids = topk_ids.contiguous()
        S = 0 if frozen else self.S
        out = self._assign(ids, S, self.stats, self.ah)
        self._copy(self.rows, min(S, ids.numel(), self.E))
        return out

    def ahead(self, ids, stats, rows):
        """K6 on predicted ids (KSTAGE_AHEAD): slots assigned and listed in cps/cpd as if the layer
        had routed them; only `rows` are copied here, the caller has the rest copied."""
        self._assign(ids, self.S, stats, 1)
        self._copy(rows, min(self.S, ids.numel(), self.E))

    def _copy(self, rows, P):
        for r in rows if P else ():
            if COPY_LIVE:
                _copy_rows_live[(COPY_LIVE,)](r, self.cps, self.cpd, r.shape[1], BS=self.BS, BLOCK=BLOCK,
                                              num_warps=WARPS)
            elif COPY_N:
                _copy_rows_few[(COPY_N,)](r, self.cps, self.cpd, P, r.shape[1], BLOCK=BLOCK, num_warps=WARPS)
            else:
                _copy_rows_kernel[(P, triton.cdiv(r.shape[1], BLOCK))](r, self.cps, self.cpd, r.shape[1],
                                                                       BLOCK=BLOCK, num_warps=WARPS)


@triton.jit
def _plan_kernel(cps_ptr, cpd_ptr, plan_ptr, done_ptr, BS: tl.constexpr):
    """KSTAGE_AHEAD: the slots K6 just assigned, as ce_helper reads a plan, [n, src0, dst0, ...]
    in rows. An empty plan is marked done here (CE_DEVDONE): the helper makes no CUDA call."""
    j = tl.arange(0, BS)
    r = tl.load(cps_ptr + j)
    ok = r >= 0
    d = tl.load(cpd_ptr + j, mask=ok, other=0)
    n = tl.sum(ok.to(tl.int32), axis=0)
    tl.store(plan_ptr + 1 + 2 * j, r.to(tl.int64), mask=ok)
    tl.store(plan_ptr + 2 + 2 * j, d.to(tl.int64), mask=ok)
    tl.store(plan_ptr + j, n.to(tl.int64) + tl.zeros([BS], tl.int64), mask=j == 0)
    tl.store(done_ptr + j, tl.full([BS], 1, tl.int32), mask=(j == 0) & (n == 0))


@triton.jit
def _ahead_kernel(x_ptr, sx, M, w_ptr, b_ptr, sc_ptr, cnt_ptr, plan_ptr, done_ptr, where_ptr, home_ptr, owner_ptr,
                  stamp_ptr, clock_ptr, freq_ptr, need_ptr, miss_ptr, cps_ptr, cpd_ptr, stats_ptr, S, slot0, pf_ptr,
                  H: tl.constexpr, E: tl.constexpr, BE: tl.constexpr, BS: tl.constexpr, BM: tl.constexpr,
                  BEB: tl.constexpr, BH: tl.constexpr, HC: tl.constexpr, K: tl.constexpr, STATS: tl.constexpr,
                  LFU: tl.constexpr, SHIFT: tl.constexpr, DRY: tl.constexpr, TH: tl.constexpr):
    """KSTAGE_AHEAD_FUSE: _Ahead.plan's eight launches (cast, gate gemv, sigmoid, bias, top-k,
    _cache_kernel AH 1, _plan_kernel) in one. Program (g, k) takes experts [g BEB, (g + 1) BEB)
    and hidden columns [k HC, (k + 1) HC) of x w.T for the M tokens of x, in fp32, into its own
    partial slot of sc (after the M x E scores): eight programs alone read the gate at ~5 GB/s
    each. The last program to finish (an atomic count, which it resets for the next replay)
    adds the partials in order (the same bits every run), takes sigmoid plus the correction
    bias into sc, each token's top K (ties: lower id), makes the ahead assign and writes the
    plan and the empty-plan flag. DRY (KSTAGE_AHEAD_DRY): the assign gets no experts, so every
    plan is empty and the cache moves as without ahead (the top-K may compile away)."""
    m = tl.arange(0, BM)
    f = tl.program_id(0) * BEB + tl.arange(0, BEB)
    k = tl.program_id(1)
    acc = tl.zeros([BM, BEB], dtype=tl.float32)
    for h0 in range(0, HC, BH):
        h = k * HC + h0 + tl.arange(0, BH)
        xv = tl.load(x_ptr + m[:, None] * sx + h[None, :], mask=(m[:, None] < M) & (h[None, :] < H), other=0.0)
        wv = tl.load(w_ptr + f[:, None] * H + h[None, :], mask=(f[:, None] < E) & (h[None, :] < H), other=0.0)
        acc += tl.dot(xv.to(tl.float32), tl.trans(wv), input_precision="ieee")
    tl.store(sc_ptr + (1 + k) * M * E + m[:, None] * E + f[None, :], acc, mask=(m[:, None] < M) & (f[None, :] < E))
    tl.debug_barrier()  # every thread's partials before the count (cooperative groups' grid sync)
    if tl.atomic_add(cnt_ptr, 1, sem="acq_rel") == tl.num_programs(0) * tl.num_programs(1) - 1:
        tl.debug_barrier()
        tl.store(cnt_ptr, 0)
        e = tl.arange(0, BE)
        mm = m < M
        mk = mm[:, None] & (e[None, :] < E)
        v = tl.zeros([BM, BE], dtype=tl.float32)
        for kk in range(0, tl.num_programs(1)):  # L2: other programs wrote them
            v += tl.load(sc_ptr + (1 + kk) * M * E + m[:, None] * E + e[None, :], mask=mk, other=0.0,
                         cache_modifier=".cg")
        v = tl.sigmoid(v) + tl.load(b_ptr + e, mask=e < E, other=0.0)[None, :]
        tl.store(sc_ptr + m[:, None] * E + e[None, :], v, mask=mk)
        v = tl.where(mk, v, float("-inf"))
        touched = tl.zeros([BE], dtype=tl.int32)
        for _ in tl.static_range(K):
            top = tl.max(v, axis=1)
            first = tl.min(tl.where(v == top[:, None], e[None, :], BE), axis=1)
            sel = (e[None, :] == first[:, None]) & mm[:, None]
            touched = tl.maximum(touched, tl.max(sel.to(tl.int32), axis=0))
            v = tl.where(sel, float("-inf"), v)
        if DRY:
            touched = touched * 0
        ok, src, dst = _k6_assign(touched, where_ptr, home_ptr, owner_ptr, stamp_ptr, clock_ptr, freq_ptr,
                                  need_ptr, miss_ptr, cps_ptr, cpd_ptr, stats_ptr, S, slot0, pf_ptr, E, BE, BS,
                                  STATS, LFU, SHIFT, 1, TH)
        j = tl.arange(0, BS)
        n = tl.sum(ok.to(tl.int32), axis=0)
        tl.store(plan_ptr + 1 + 2 * j, src.to(tl.int64), mask=ok)
        tl.store(plan_ptr + 2 + 2 * j, dst.to(tl.int64), mask=ok)
        tl.store(plan_ptr + j, n.to(tl.int64) + tl.zeros([BS], tl.int64), mask=j == 0)
        tl.store(done_ptr + j, tl.full([BS], 1, tl.int32), mask=(j == 0) & (n == 0))


def fused_scratch(M, E, H):
    """Floats of sc _ahead_kernel needs for up to M tokens: the scores, then a partial a chunk."""
    return M * E * (1 + triton.cdiv(H, AHEAD_HC))


def fused_plan(c, x, w, bias, sc, cnt, plan, done, stats, rows):
    """_ahead_kernel on cache c, then the small rows' copies (c.ahead's)."""
    x = x.contiguous()
    M, H = x.shape
    _ahead_kernel[(triton.cdiv(c.E, AHEAD_BEB), triton.cdiv(H, AHEAD_HC))](
        x, x.stride(0), M, w, bias, sc, cnt, plan, done, c.where, c.home, c.owner, c.stamp, c.clock, c.freq,
        c.need, c.miss, c.cps, c.cpd, stats if stats is not None else c.clock, c.S, c.slot0, c.pf, H=H, E=c.E,
        BE=c.BE, BS=c.BS, BM=max(16, triton.next_power_of_2(M)), BEB=AHEAD_BEB, BH=AHEAD_BH,
        HC=AHEAD_HC, K=AHEAD_K, STATS=stats is not None, LFU=c.lfu, SHIFT=c.shift, DRY=AHEAD_DRY, TH=c.stale,
        num_warps=AHEAD_WARPS)
    c._copy(rows, min(c.S, M * AHEAD_K, c.E))


@triton.jit
def _tick_kernel(t0_ptr, st_ptr, plan_ptr, END: tl.constexpr):
    """KSTAGE_AHEAD_TICK (B6): %globaltimer just before _Ahead.wait's flag wait (END False) and
    just after its reset (END True), which adds the gap to the layer's tally: waits, ns, waits
    on a plan with copies, their ns, max ns, waits over 20 us. One writer a layer: plain stores."""
    t = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
    if END:
        d = t - tl.load(t0_ptr)
        f = (tl.load(plan_ptr, volatile=True) > 0).to(tl.int64)
        tl.store(st_ptr, tl.load(st_ptr) + 1)
        tl.store(st_ptr + 1, tl.load(st_ptr + 1) + d)
        tl.store(st_ptr + 2, tl.load(st_ptr + 2) + f)
        tl.store(st_ptr + 3, tl.load(st_ptr + 3) + f * d)
        tl.store(st_ptr + 4, tl.maximum(tl.load(st_ptr + 4), d))
        tl.store(st_ptr + 5, tl.load(st_ptr + 5) + (d > 20000).to(tl.int64))
    else:
        tl.store(t0_ptr, t)


class _Ahead:
    """KSTAGE_AHEAD=1, steps of fewer than KSTAGE_AHEAD_M tokens and at least KSTAGE_AHEAD_MIN (decode).
    At MoE layer p the next MoE layer's gate, scaled by the two layers' norm weights (_install_predict's "d1
    scaled"), is applied to layer p's input, and K6 assigns layer p+1's slots to the top
    KSTAGE_AHEAD_K predicted ids as if layer p+1 had routed them. The small rows are copied
    here; the big ones (host pages) by ce_helper on the copy engine, while layer p and the
    layers between run. Layer p+1 waits for those copies (cuStreamWaitValue32 on a flag the
    helper writes), then takes its own K6 step, which copies what the prediction missed.
    Plan and flags are memory, not graph edges, so this works inside CUDA graphs and across
    vLLM's piecewise graph boundaries. The helper thread spins on its flags (one CPU core).
    KSTAGE_AHEAD_SIDE=1 runs the predictor and plan on a side stream beside layer p's expert
    compute (joined before the MoE call returns); KSTAGE_AHEAD_STALE=U lets a fill evict only
    empty slots and experts whose LFU count is below U uses' worth."""

    def start(self, order, model, period):
        """order: (idx, name, cache, big rows, small rows) for every MoE layer, in forward order."""
        import faulthandler, signal, subprocess, threading
        faulthandler.register(signal.SIGUSR1, all_threads=True)  # CE_WATCH: every thread's stack
        mods = dict(model.named_modules())
        self.L = L = len(order)
        self.side = torch.cuda.Stream() if AHEAD_SIDE else None
        self.caches, self.small = [o[2] for o in order], [o[4] for o in order]
        norms, self.w, self.bias = [], [None], [None]
        for p, (idx, name, *_) in enumerate(order):
            gate = mods[name.rsplit(".", 1)[0]].gate
            assert gate.e_score_correction_bias is not None, f"{name}: no correction bias"
            norms.append(mods[name.rsplit(".", 2)[0]].norm.weight.detach().float())
            if p:
                self.w.append((gate.weight.detach().float() * (norms[p] / norms[p - 1])).contiguous())
                self.bias.append(gate.e_score_correction_bias.detach().float())
        nt = {len(o[3]) for o in order}
        assert len(nt) == 1 and min(nt), f"big tensors per layer: {nt}"
        self.ntens = nt.pop()
        maxk = max(c.BS for c in self.caches)
        so = "/tmp/ce_helper.so"
        if "helper" not in AHEAD_SKIP:
            subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-I/usr/local/cuda/include",
                            f"{os.path.dirname(os.path.abspath(__file__))}/ce_helper.c", "-o", so,
                            "-L/usr/local/cuda/lib64/stubs", "-lcuda", "-lpthread"], check=True)
            lib = ctypes.CDLL(so)
            lib.ce_start.restype = ctypes.c_void_p
            lib.ce_start.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 6
        self.cu = ctypes.CDLL("libcuda.so.1")
        for f in ("cuStreamWriteValue32_v2", "cuStreamWaitValue32_v2"):
            getattr(self.cu, f).argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
            getattr(self.cu, f).restype = ctypes.c_int
        w = 1 + 2 * maxk
        self.plan_h = torch.zeros(L, w, dtype=torch.int64, pin_memory=True)
        self.plan_d = torch.as_tensor(_Arr(self.plan_h.data_ptr(), self.plan_h.numel() * 8),
                                      device="cuda").view(torch.int64).view(L, w)
        self.req = torch.zeros(L, dtype=torch.int32, pin_memory=True)
        # A VMM page, not torch memory: the helper writes it from a context of its own (CE_OWNCTX)
        self.done_m = MixedRows(1, 4 * L, 1)
        self.done = self.done_m.tensor(torch.int32, (L,))
        self.done.zero_()
        base = [r.data_ptr() for o in order for r in o[3]]
        rb = [r.shape[1] * r.element_size() for o in order for r in o[3]]
        B = ctypes.c_uint64 * len(base)
        ctx = ctypes.c_void_p()
        assert self.cu.cuCtxGetCurrent(ctypes.byref(ctx)) == 0 and ctx.value
        os.environ["CE_DEVDONE"], os.environ["CE_DTOD"] = "1", "1"
        os.environ.setdefault("CE_OWNCTX", "1")  # on vLLM's context the helper deadlocks (ce_helper.c)
        os.environ.setdefault("CE_WATCH", "30")  # a helper stuck 30 s in one CUDA call ends the process
        torch.cuda.synchronize()
        if "helper" not in AHEAD_SKIP:
            self.ce = lib.ce_start(ctx, L, self.ntens, maxk, ctypes.c_void_p(self.plan_h.data_ptr()),
                                   ctypes.c_void_p(self.req.data_ptr()), ctypes.c_void_p(self.done.data_ptr()),
                                   B(*base), B(*base), B(*rb))
            self.lib = lib
        self.hstats, self.dstats = _stats_buffer(L, 6) if period else (None, None)
        if AHEAD_FUSE:  # _ahead_kernel's scores (layers run one at a time) and per-layer counts
            n = fused_scratch(AHEAD_M, max(c.E for c in self.caches), max(w.shape[1] for w in self.w if w is not None))
            self.sc = torch.empty(n, dtype=torch.float32, device="cuda")
            self.cnt = torch.zeros(L, dtype=torch.int32, device="cuda")
        for c in self.caches:
            c.ah = 0 if "ah" in AHEAD_SKIP else 2
            c.stale = round(AHEAD_STALE * 2**20)
        if period:
            _report(self.hstats, period, "ahead")
            if "helper" not in AHEAD_SKIP:
                threading.Thread(target=self._counts, args=(period,), daemon=True, name="kstage-ce").start()
        if AHEAD_TICK:
            self.t0 = torch.zeros(L, dtype=torch.int64, device="cuda")
            self.th, self.td = _stats_buffer(L, 6)
            threading.Thread(target=self._ticks, args=(period or 30,), daemon=True, name="kstage-tick").start()

    def _ticks(self, period):
        """KSTAGE_AHEAD_TICK's tally: the time the stream spends in wait(p) for p = 1 .. L - 1,
        split by whether layer p - 1 planned copies (the helper writes the flag) or not."""
        last = None
        while True:
            time.sleep(period)
            v = self.th.sum(0).tolist()
            if v != last and v[0]:
                e, steps = v[0] - v[2], v[0] / (self.L - 1)
                log(f"ahead waits: {v[0]}, {v[1] / steps / 1e3:.0f} us a step; empty plan "
                    f"{(v[1] - v[3]) / max(e, 1) / 1e3:.1f} us over {e}, with copies {v[3] / max(v[2], 1) / 1e3:.1f} "
                    f"us over {v[2]}; max {int(self.th[:, 4].max()) / 1e3:.0f} us, {v[5]} over 20 us")
            last = v

    def _counts(self, period):
        """The helper's own tally (ce_counts makes no CUDA call): its rows must match the "ahead:"
        line's copies. A failed copy ends the process (CE_WATCH), so err stays 0 here."""
        v, last = (ctypes.c_int64 * 4)(), None
        while True:
            time.sleep(period)
            self.lib.ce_counts(ctypes.c_void_p(self.ce), v)
            if list(v) != last and v[0]:
                log(f"ahead helper: {v[0]} requests, {v[1]} rows, {v[3]} copies, err {v[2]}")
            last = list(v)

    def _memop(self, f, addr, v, flags=0):
        r = getattr(self.cu, f)(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), addr, v, flags)
        assert r == 0, f"{f}: CUresult {r}"

    def wait(self, p):
        """Before layer p's own K6 step: the copies planned at layer p - 1 are done."""
        if p and "wait" not in AHEAD_SKIP:
            a = self.done.data_ptr() + 4 * p
            if AHEAD_TICK:
                _tick_kernel[(1,)](self.t0[p:], self.td[p], self.plan_d[p], END=False)
            self._memop("cuStreamWaitValue32_v2", a, 1, 1)  # CU_STREAM_WAIT_VALUE_EQ
            self._memop("cuStreamWriteValue32_v2", a, 0)
            if AHEAD_TICK:
                _tick_kernel[(1,)](self.t0[p:], self.td[p], self.plan_d[p], END=True)

    def plan(self, p, x):
        """After layer p's own K6 step (its copies come first): layer p + 1's fill from x. With
        KSTAGE_AHEAD_SIDE, on a side stream beside layer p's expert compute until join(); the
        expert kernels only read x (vLLM's modular kernel allocates its own output)."""
        if p + 1 == self.L:
            return
        if self.side is None:
            return self._plan(p, x)
        self.side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.side):
            self._plan(p, x)

    def join(self):
        """After layer p's expert compute: the side stream rejoins (a graph's forks join in it)."""
        if self.side is not None:
            torch.cuda.current_stream().wait_stream(self.side)

    def _plan(self, p, x):
        q = p + 1
        c = self.caches[q]
        st = self.dstats[q] if self.dstats is not None else None
        if "pred" in AHEAD_SKIP:
            if "wait" not in AHEAD_SKIP:
                self.done[q].fill_(1)
        elif AHEAD_FUSE and x.shape[0] <= AHEAD_FUSE_M:
            fused_plan(c, x, self.w[q], self.bias[q], self.sc, self.cnt[q:], self.plan_d[q], self.done[q:], st,
                       self.small[q])
        else:
            lg = x.float() @ self.w[q].t()
            ids = (lg.sigmoid() + self.bias[q]).topk(AHEAD_K, dim=-1).indices
            if AHEAD_DRY:  # the predictor alone: an empty plan, its flag set here
                self.plan_d[q, 0].zero_()
                self.done[q].fill_(1)
            else:
                c.ahead(ids, st, self.small[q])
                _plan_kernel[(1,)](c.cps, c.cpd, self.plan_d[q], self.done[q:], BS=c.BS)
        if "req" not in AHEAD_SKIP:
            self._memop("cuStreamWriteValue32_v2", self.req.data_ptr() + 4 * q, 1)


class _BigStage:
    """K6 + KSTAGE_DMA_M. A batch that big (a prefill chunk) routes to nearly every expert, far
    more than the free slots, so K6 leaves the rest at home and Marlin reads them through UVA,
    once per block of tokens routed to them. Instead: at such a step's first MoE layer, read
    every layer's rows (one sync); copy each layer's non-resident experts, home order, into
    staging rows [Z, Z + n) with the copy engine on a side stream, KSTAGE_DMA_BUF layers ahead
    (default 2: the next layer copies while this one computes); remap ids through that map.
    The copy needs no routing, so it can run ahead; staging is one set of device pages per
    buffer, mapped into every layer's range. The step leaves the cache alone (no stamps, no
    slot copies): a chunk that touches every expert says nothing about the next decode step."""

    def __init__(self, nbuf):
        self.nbuf, self.layers, self.side = nbuf, [], torch.cuda.Stream()
        self.free = [None] * nbuf
        self.next, self.steps, self.runs, self.rows = 0, 0, 0, 0
        self.time = os.environ.get("KSTAGE_DMA_TIME", "0") == "1"  # time steps (syncs at the last layer)

    def add(self, cache, D, Z, n, big, small):
        """One layer, in forward order: big are the [rows, units] views the copy engine fills,
        small the device-only ones (filled with an index_select)."""
        E = cache.E
        self.layers.append({"c": cache, "D": D, "Z": Z, "n": n, "big": big, "small": small, "E": E,
                            "pin": torch.empty(E + n, dtype=torch.long, pin_memory=True),
                            "dev": torch.empty(E + n, dtype=torch.long, device="cuda"),
                            "done": torch.cuda.Event(enable_timing=self.time), "runs": [],
                            "c0": torch.cuda.Event(enable_timing=True) if self.time else None,
                            "bytes": n * sum(r.shape[1] * r.element_size() for r in big)})

    def _begin(self, li0, m):
        snap = torch.stack([L["c"].where for L in self.layers]).cpu()  # waits for the layers before
        self.m = m
        if self.time:
            self.f0 = torch.cuda.Event(enable_timing=True); self.f0.record()
        for L, w in zip(self.layers, snap.long()):
            nr = (w >= L["D"]).nonzero().flatten()
            nr = nr[torch.argsort(w[nr])]
            src = w[nr]
            assert src.numel() == L["n"], (src.numel(), L["n"])
            w[nr] = L["Z"] + torch.arange(L["n"])
            L["pin"][:L["E"]] = w
            L["pin"][L["E"]:] = src
            s, runs, i = src.tolist(), [], 0
            while i < len(s):
                j = i
                while j + 1 < len(s) and s[j + 1] == s[j] + 1:
                    j += 1
                runs.append((s[i], L["Z"] + i, j - i + 1))
                i = j + 1
            L["runs"] = runs
            self.runs += len(runs); self.rows += len(s)
        self.steps += 1
        if self.steps in (1, 2, 3, 10, 100) or self.steps % 1000 == 0:
            log(f"dma: {self.steps} big steps, {self.rows / self.steps:.0f} rows in "
                f"{self.runs / self.steps:.0f} runs a step")
        for li in range(li0, min(li0 + self.nbuf, len(self.layers))):
            self._copy(li)

    def _copy(self, li):
        L, b = self.layers[li], li % self.nbuf
        with torch.cuda.stream(self.side):
            if self.free[b] is not None:
                self.side.wait_event(self.free[b])  # the layer that read buffer b is done
            if self.time:
                L["c0"].record(self.side)
            for r in L["big"]:
                for s, d, n in L["runs"]:
                    r[d:d + n].copy_(r[s:s + n], non_blocking=True)
            L["done"].record(self.side)

    def enter(self, li, ids):
        if li != self.next or li == 0:
            self._begin(li, ids.shape[0])
        L = self.layers[li]
        torch.cuda.current_stream().wait_event(L["done"])
        L["dev"].copy_(L["pin"], non_blocking=True)
        E, Z, n = L["E"], L["Z"], L["n"]
        for r in L["small"]:
            r[Z:Z + n].copy_(r.index_select(0, L["dev"][E:]))
        return L["dev"][:E].index_select(0, ids.reshape(-1).long()).view(ids.shape).to(ids.dtype)

    def leave(self, li):
        b = li % self.nbuf
        if self.free[b] is None:
            self.free[b] = torch.cuda.Event()
        self.free[b].record()
        self.next = (li + 1) % len(self.layers)
        if li + self.nbuf < len(self.layers):
            self._copy(li + self.nbuf)
        elif self.time and li == len(self.layers) - 1:
            f1 = torch.cuda.Event(enable_timing=True); f1.record(); f1.synchronize()
            busy = sum(L["c0"].elapsed_time(L["done"]) for L in self.layers)
            gib = sum(L["bytes"] for L in self.layers) / 2**30
            log(f"dma step {self.steps} ({self.m} tokens): copy engine busy {busy:.1f} ms for {gib:.2f} GiB "
                f"({gib * 2**30 / busy / 1e6:.1f} GB/s); MoE layers span {self.f0.elapsed_time(f1):.1f} ms")


# ---------------------------------------------------------------- finding the expert tensors

def _walk(roots, depth=4):
    """(path, tensor) reachable from (name, object) roots through vLLM objects' attributes,
    without walking into other nn.Modules (the quant config keeps a back-reference to the layer)."""
    seen, out, stack = set(), [], [(r, n, 0) for n, r in roots]
    while stack:
        o, path, d = stack.pop()
        if o is None or id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, torch.Tensor):
            out.append((path, o))
            continue
        if d > depth or (isinstance(o, torch.nn.Module) and d > 0):
            continue
        if isinstance(o, (list, tuple)):
            items = [(str(i), x) for i, x in enumerate(o)]
        elif isinstance(o, dict):
            items = [(str(k), v) for k, v in o.items()]
        elif isinstance(o, torch.nn.Module):
            items = list(o._parameters.items()) + list(o._buffers.items()) + [
                (k, v) for k, v in vars(o).items() if k not in ("_parameters", "_buffers", "_modules")]
        elif hasattr(o, "__dict__") and type(o).__module__.startswith("vllm"):
            items = list(vars(o).items())
        else:
            continue
        stack += [(x, f"{path}.{k}", d + 1) for k, x in items]
    return out


SKIP_NAMES = ("workspace", "correction_bias", "input_scale", "expert_map", "hash_indices")


def _groups(runner):
    """Tensors behind this layer's routed experts, grouped by the memory they view:
    [(nbytes, [(path, tensor), ...]), ...]. Anything the router or gate can reach is left out,
    and so are names in SKIP_NAMES: routing runs on the original expert ids (the score-correction
    bias is expert-indexed and held by both the router and the experts)."""
    re_ = runner.routed_experts
    qm = re_.quant_method
    mk = getattr(qm, "moe_kernel", None)
    fe = getattr(mk, "fused_experts", None)
    skip = {t.data_ptr() for _, t in _walk([("router", runner.router), ("gate", getattr(runner, "gate", None))])
            if t.numel()}
    roots = [("experts", re_), ("qm", qm), ("qcfg", getattr(qm, "moe_quant_config", None)), ("kernel", mk),
             ("fused", fe), ("fused.qcfg", getattr(fe, "quant_config", None))]
    g = {}
    for path, t in _walk(roots):
        if t.numel() == 0 or t.data_ptr() in skip or any(s in path for s in SKIP_NAMES):
            continue
        g.setdefault((t.data_ptr(), t.numel() * t.element_size()), {})[id(t)] = (path, t)
    return [(k[1], list(v.values())) for k, v in g.items()]


def _show(name, gs, E):
    """Log the tensors found for one layer, once, so a wrong walk is visible."""
    if getattr(_show, "done", False) and not os.environ.get("KSTAGE_LOG"):
        return
    _show.done = True
    for nb, grp in sorted(gs, key=lambda g: -g[0]):
        t = grp[0][1]
        log(f"{name}: {nb / MiB:9.2f} MiB {str(tuple(t.shape)):22} {str(t.dtype):14} "
            f"{'host' if is_host(t) else 'dev '} {'expert-major' if t.dim() and t.shape[0] == E else '-':12} "
            + ", ".join(p for p, _ in grp))


def _set(group, new):
    """Point every tensor object in the group at new, each keeping its own dtype and shape."""
    for _, t in group:
        t.data = new if (t.dtype, t.shape) == (new.dtype, new.shape) else new.view(t.dtype).view(t.shape)


def _layers(model):
    out = []
    for name, m in model.named_modules():
        if hasattr(m, "router") and hasattr(m, "routed_experts"):
            assert not m.routed_experts.quant_method.is_monolithic, f"{name}: monolithic MoE kernel"
            mm = re.search(r"layers\.(\d+)\.", name + ".")
            out.append((int(mm.group(1)) if mm else len(out), name, m))
    assert len({id(m.router) for _, _, m in out}) == len(out), "routers shared between layers"
    return out


# ---------------------------------------------------------------- the two modes

DMA_M = int(os.environ.get("KSTAGE_DMA_M", "0"))


class _Staging:
    """One device buffer per (shape, dtype), shared by every layer: layers run one at a time."""

    def __init__(self):
        self.bufs = {}

    def get(self, t):
        k = (tuple(t.shape), t.dtype)
        if k not in self.bufs:
            self.bufs[k] = torch.empty(t.shape, dtype=t.dtype, device="cuda")
        return self.bufs[k]

    def nbytes(self):
        return sum(b.numel() * b.element_size() for b in self.bufs.values())


def _is_big(t, E):
    return t.dim() >= 1 and t.shape[0] == E and t.is_contiguous() and t.numel() * t.element_size() >= MiB


def _install_gather(layers):
    st, n_layers, moved = _Staging(), 0, 0
    for idx, name, runner in layers:
        E = runner.routed_experts.w13_weight.shape[0]
        big = []  # (source view, host alias, staging, holders)
        gs = _groups(runner)
        _show(name, gs, E)
        for nb, grp in gs:
            t0 = grp[0][1]
            if not is_host(t0):
                continue
            if _is_big(t0, E):
                src = t0.data
                stg = st.get(src)
                big.append((src, host_alias(src) if DMA_M else None, stg, grp))
                _set(grp, stg)
                moved += nb
            else:  # scales, alphas: small, keep a device copy
                _set(grp, t0.data.clone())
        if not big:
            continue
        n_layers += 1
        _keep.append(big)
        _wrap(runner, idx, E, big=big)
    log(f"gather: {n_layers} layers, {moved / 2**30:.2f} GiB of experts staged through "
        f"{st.nbytes() / 2**30:.2f} GiB of device buffers; DMA from {DMA_M or 'never'} tokens")


def _wrap(runner, idx, E, big=(), pos=None, mixed=(), cache=None, stage=None, ahead=None):
    re_ = runner.routed_experts
    fwd = re_.forward_modular

    def forward_modular(x, topk_weights, topk_ids, *a, **k):
        _lora_ids[0] = topk_ids  # adapters are indexed by the router's ids (_lora_fix)
        try:
            return remapped(x, topk_weights, topk_ids, *a, **k)
        finally:
            _lora_ids[0] = None

    def remapped(x, topk_weights, topk_ids, *a, **k):
        M = x.shape[0]
        if cache is not None and stage and M >= DMA_M and not torch.cuda.is_current_stream_capturing():
            st, li = stage  # K6 prefill: non-resident experts staged on the copy engine
            ids = st.enter(li, topk_ids)
            try:
                return fwd(x, topk_weights, ids, *a, **k)
            finally:
                st.leave(li)
        if cache is not None:  # K6: slots filled, ids remapped to rows
            # _Ahead: one forward has one M, so every layer agrees. Only in graphs, where decode runs;
            # eager steps of this size are warm-up and profiling.
            fill = ahead is not None and AHEAD_MIN <= M < AHEAD_M and torch.cuda.is_current_stream_capturing()
            if fill:
                ahead[0].wait(ahead[1])
            ids = cache.step(topk_ids, bool(FREEZE_M) and M >= FREEZE_M)
            if fill:
                ahead[0].plan(ahead[1], x)
            out = fwd(x, topk_weights, ids, *a, **k)
            if fill:
                ahead[0].join()
            return out
        ids = topk_ids if pos is None else pos[topk_ids.long()].to(topk_ids.dtype)
        if big:  # K2: experts live in host memory, staging is what Marlin reads
            if DMA_M and M >= DMA_M:
                for src, alias, stg, _ in big:
                    stg.copy_(alias, non_blocking=True)
            else:
                order = plan(ids, E)
                for src, _, stg, _ in big:
                    gather(src, stg, order, ids.numel())
            return fwd(x, topk_weights, ids, *a, **k)
        if mixed and DMA_M and M >= DMA_M:  # K5 prefill: copy the layer whole, run, swap back
            for mt, stg, grp in mixed:
                stg.copy_(mt, non_blocking=True)
                _set(grp, stg)
            try:
                return fwd(x, topk_weights, ids, *a, **k)
            finally:
                for mt, stg, grp in mixed:
                    _set(grp, mt)
        return fwd(x, topk_weights, ids, *a, **k)

    re_.forward_modular = forward_modular


def _profile(layers, E):
    path = os.environ.get("KSTAGE_PROFILE")
    if not path:
        return {idx: [0] * E for idx, _, _ in layers}, "none (expert index order)"
    d = json.load(open(path))
    c = d["counts"]
    missing = [idx for idx, _, _ in layers if str(idx) not in c]
    assert not missing, f"profile {path} has no counts for decoder layers {missing}"
    return {idx: c[str(idx)] for idx, _, _ in layers}, f"{path} ({d.get('kind', '?')})"


def _cold_split(layers, groups, E):
    """How many experts of each layer go to host memory: KSTAGE_COLD_GB, KSTAGE_POLICY."""
    big = lambda g: _is_big(g[0][1], E)
    row = {idx: sum(nb // E for nb, g in gs if big(g)) for idx, gs in groups.items()}
    host_now = sum(nb for gs in groups.values() for nb, g in gs if big(g) and is_host(g[0][1]))
    gb = os.environ.get("KSTAGE_COLD_GB")
    budget = float(gb) * 2**30 if gb else host_now
    rb = max(row.values())
    n_cold = min(int(budget // rb), E * len(layers))
    counts, src = _profile(layers, E)
    pol = os.environ.get("KSTAGE_POLICY", "global")
    cold_n = {}
    if pol == "static":
        per, extra = divmod(n_cold, len(layers))
        for i, (idx, _, _) in enumerate(layers):
            cold_n[idx] = per + (1 if i < extra else 0)
    else:  # least-routed (layer, expert) pairs anywhere; ties go to the higher index
        pairs = sorted((counts[idx][e], -idx, -e, idx) for idx, _, _ in layers for e in range(E))
        for idx, _, _ in layers:
            cold_n[idx] = 0
        for *_, idx in pairs[:n_cold]:
            cold_n[idx] += 1
    return cold_n, counts, src, pol, n_cold


def _install_hotcold(layers):
    E = layers[0][2].routed_experts.w13_weight.shape[0]
    groups = {idx: _groups(r) for idx, _, r in layers}
    _show(layers[0][1], groups[layers[0][0]], E)
    big = lambda g: _is_big(g[0][1], E)
    cold_n, counts, src, pol, n_cold = _cold_split(layers, groups, E)
    st = _Staging() if DMA_M else None
    dev_bytes = host_bytes = 0
    # layers on the card first: they give back device memory before offloaded layers take it
    on_host = {idx: any(big(g) and is_host(g[0][1]) for _, g in groups[idx]) for idx, _, _ in layers}
    for idx, name, runner in sorted(layers, key=lambda l: (on_host[l[0]], l[0])):
        cnt = torch.tensor(counts[idx], dtype=torch.float64)
        perm = sorted(range(E), key=lambda e: (-cnt[e].item(), e))  # hot first
        perm_t = torch.tensor(perm, dtype=torch.long, device="cuda")
        pos = torch.empty(E, dtype=torch.long, device="cuda")
        pos[perm_t] = torch.arange(E, device="cuda")
        n_hot = E - cold_n[idx]
        mixed = []
        for nb, grp in groups[idx]:
            t0 = grp[0][1]
            if t0.dim() < 1 or t0.shape[0] != E:
                if is_host(t0):
                    _set(grp, t0.data.clone())
                continue
            if not _is_big(t0, E):
                _set(grp, t0.data.index_select(0, perm_t).contiguous())
                continue
            old = t0.data
            mx = MixedRows(E, nb // E, n_hot)
            mt = mx.tensor(old.dtype, old.shape)
            for s in range(0, E, 16):
                mt[s:s + 16].copy_(old.index_select(0, perm_t[s:s + 16]))
            _keep.append(mx)
            _set(grp, mt)
            dev_bytes += mx.dsize; host_bytes += mx.hsize
            mixed.append((mt, st.get(mt) if st else None, grp))
            del old
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _keep.append(pos)
        _wrap(runner, idx, E, pos=pos, mixed=mixed)
    if hasattr(torch._C, "_host_emptyCache"):
        torch._C._host_emptyCache()  # the pinned blocks vLLM's offloader left behind
    log(f"hotcold: {len(layers)} layers, {n_cold} experts cold ({n_cold / len(layers):.1f} a layer, "
        f"{host_bytes / 2**30:.2f} GiB host pages, {dev_bytes / 2**30:.2f} GiB device), policy {pol}, "
        f"profile {src}; cold per layer {[cold_n[i] for i, _, _ in layers]}"
        + (f"; DMA staging {st.nbytes() / 2**30:.2f} GiB from {DMA_M} tokens" if st else ""))


SLOTS = os.environ.get("KSTAGE_SLOTS", "all")
FREEZE_M = int(os.environ.get("KSTAGE_FREEZE_M", "0"))
EVICT = os.environ.get("KSTAGE_EVICT", "lru")
SHIFT = int(os.environ.get("KSTAGE_SHIFT", "6"))
AHEAD = os.environ.get("KSTAGE_AHEAD", "0") == "1"  # _Ahead: fill the next layer's slots on the copy engine
AHEAD_M = int(os.environ.get("KSTAGE_AHEAD_M", "64"))  # steps of fewer tokens fill ahead (decode)
AHEAD_MIN = int(os.environ.get("KSTAGE_AHEAD_MIN", "1"))  # B4: and of at least this many (a graph's M is fixed)
AHEAD_K = int(os.environ.get("KSTAGE_AHEAD_K", "6"))  # ids predicted per token
AHEAD_FUSE = os.environ.get("KSTAGE_AHEAD_FUSE", "0") == "1"  # the predictor in one launch (_ahead_kernel)
AHEAD_DRY = os.environ.get("KSTAGE_AHEAD_DRY", "0") == "1"  # predict, plan no copies: ahead's fixed cost
AHEAD_TICK = os.environ.get("KSTAGE_AHEAD_TICK", "0") == "1"  # time the flag waits (_tick_kernel)
AHEAD_SIDE = os.environ.get("KSTAGE_AHEAD_SIDE", "0") == "1"  # p30: the predictor beside layer p's experts
# p30: an ahead fill evicts only experts whose LFU count is below this many uses' worth (a use
# adds 2**20; counts decay by count >> KSTAGE_SHIFT a step). 0 = any victim K6 would take.
AHEAD_STALE = float(os.environ.get("KSTAGE_AHEAD_STALE", "0"))
assert not AHEAD_STALE or EVICT == "lfu", "KSTAGE_AHEAD_STALE reads LFU counts: KSTAGE_EVICT=lfu"
# p26: what DRY's fixed cost is made of. Each part named is left out: pred (the predictor; a fill
# sets the flag if a wait reads it), req (the request flag's memop), wait (the flag wait and its
# reset), ah (K6's real step as without ahead), helper (ce_helper never starts: no context of its own)
AHEAD_SKIP = set(filter(None, os.environ.get("KSTAGE_AHEAD_SKIP", "").split(",")))
assert AHEAD_SKIP <= {"pred", "req", "wait", "ah", "helper"}, f"KSTAGE_AHEAD_SKIP: {AHEAD_SKIP}"
assert not AHEAD_SKIP or AHEAD_DRY, "KSTAGE_AHEAD_SKIP leaves out parts copies need: KSTAGE_AHEAD_DRY=1 only"
assert "ah" not in AHEAD_SKIP or "pred" in AHEAD_SKIP, "the predictor's assign expects ah 2"
assert "helper" not in AHEAD_SKIP or "req" in AHEAD_SKIP, "requests need the helper"
AHEAD_BEB = int(os.environ.get("KSTAGE_AHEAD_BEB", "16"))  # _ahead_kernel: experts a program scores
AHEAD_BH = int(os.environ.get("KSTAGE_AHEAD_BH", "128"))  # _ahead_kernel: hidden columns a load
AHEAD_HC = int(os.environ.get("KSTAGE_AHEAD_HC", "384"))  # _ahead_kernel: hidden columns a program (a BH multiple)
AHEAD_WARPS = int(os.environ.get("KSTAGE_AHEAD_WARPS", "4"))  # _ahead_kernel: warps a program
# Fused up to this many tokens: above 16, torch's gate gemm is a faster kernel (ahead_fused_test.py)
AHEAD_FUSE_M = int(os.environ.get("KSTAGE_AHEAD_FUSE_M", "16"))


def _set_rows(group, new):
    """_set for a tensor with a different number of expert rows: every holder is expert-major."""
    for path, t in group:
        assert t.dim() and t.shape[0] == group[0][1].shape[0], f"{path}: {tuple(t.shape)} is not expert-major"
        t.data = new.view(t.dtype).view((new.shape[0],) + tuple(t.shape[1:]))


def _stats_buffer(n, *shape):
    """n x 3 (or n x shape) int64 in pinned host memory, with a device view of the same bytes:
    the kernel writes, a thread reads without a CUDA call (none may run while vLLM captures graphs)."""
    shape = (n,) + (shape or (3,))
    host = torch.zeros(shape, dtype=torch.int64, pin_memory=True)
    dev = torch.as_tensor(_Arr(host.data_ptr(), host.numel() * 8), device="cuda").view(torch.int64).view(shape)
    return host, dev


def _report(host, period, name="cache"):
    import threading

    def run():
        last = None
        while True:
            time.sleep(period)
            v = host.sum(0).tolist()
            if v != last and v[0]:
                log(f"{name}: {v[0]} layer-steps, {v[1] / v[0]:.2f} misses and {v[2] / v[0]:.2f} copies "
                    f"a layer-step ({v[1]} / {v[2]})" + ("" if len(v) == 3 else f"; evicted unused fills {v[5]}"
                    if name == "ahead" else f"; ahead fills used {v[3]}, evicted unused {v[5]}, "
                    f"misses of experts the fill evicted {v[4]}"))
            last = v
    threading.Thread(target=run, daemon=True, name="kstage-stats").start()


def _cache_layout(cnt, D, S):
    """One layer's K6 rows: [0, H) the H = D - S most routed experts, pinned; [H, D) slots,
    starting with the next S; [D, E + S) a host home for every unpinned expert, hot first.
    src: the expert each row starts as. freq: LFU's starting counts, the profile's order (all
    below one use). Ties as expert_cache_sim.py (lower id is colder)."""
    E = len(cnt)
    H = D - S
    perm = sorted(range(E), key=lambda e: (-cnt[e], -e))
    unpinned = perm[H:]
    where, home, freq = [0] * E, [-1] * E, [0] * E
    for i, e in enumerate(perm):
        freq[e] = E - i
    for i, e in enumerate(perm[:H]):
        where[e] = i
    for k, e in enumerate(unpinned):
        home[e] = D + k
        where[e] = H + k if k < S else D + k
    return {"H": H, "S": S, "D": D, "where": where, "home": home, "owner": perm[H:D],
            "stamp": [-1 - j for j in range(S)], "freq": freq, "src": perm[:D] + unpinned}


def _install_cache(layers, model=None):
    E = layers[0][2].routed_experts.w13_weight.shape[0]
    groups = {idx: _groups(r) for idx, _, r in layers}
    _show(layers[0][1], groups[layers[0][0]], E)
    big = lambda g: _is_big(g[0][1], E)
    cold_n, counts, src, pol, n_cold = _cold_split(layers, groups, E)
    period = float(os.environ.get("KSTAGE_STATS", "0"))
    hstats, dstats = _stats_buffer(len(layers), 6 if AHEAD else 3) if period else (None, None)
    dev_bytes = host_bytes = 0
    shape = []
    on_host = {idx: any(big(g) and is_host(g[0][1]) for _, g in groups[idx]) for idx, _, _ in layers}
    X = max(cold_n.values()) if DMA_M else 0  # staging rows: the most experts any layer has at home
    fpos = {idx: i for i, idx in enumerate(sorted(i for i, _, _ in layers))}  # forward order
    nbuf = int(os.environ.get("KSTAGE_DMA_BUF", "2"))
    bstage, staged, bank, bank_bytes = (_BigStage(nbuf) if X else None), {}, {}, 0
    G = _vmm()[3] if X else 1
    up = lambda x: (x + G - 1) // G * G
    ahead, rowsets = (_Ahead() if AHEAD else None), {}
    for li, (idx, name, runner) in enumerate(sorted(layers, key=lambda l: (on_host[l[0]], l[0]))):
        D = E - cold_n[idx]
        lay = _cache_layout(counts[idx], D, D if SLOTS == "all" else min(int(SLOTS), D))
        c = _Cache(lay, dstats[li] if dstats is not None else None, evict=EVICT, shift=SHIFT)
        H, S, R = lay["H"], lay["S"], E + lay["S"]
        # staging rows start at Z, the first row whose page is past every big tensor's host pages
        Z = max([-(-up(R * (nb // E)) // (nb // E)) for nb, g in groups[idx] if big(g)] or [R])
        T = Z + X if X else R
        bigv, smallv = [], []
        src_rows = torch.tensor(lay["src"], dtype=torch.long, device="cuda")
        for nb, grp in groups[idx]:
            t0 = grp[0][1]
            if t0.dim() < 1 or t0.shape[0] != E:
                if is_host(t0):
                    _set(grp, t0.data.clone())
                continue
            old, isbig = t0.data, _is_big(t0, E)  # before _set_rows: it makes t0 T rows
            if isbig:
                rb, stage = nb // E, None
                if X:  # one set of device pages per tensor kind and buffer, shared by every layer
                    key = (min(p for p, _ in grp), fpos[idx] % nbuf)
                    if key not in bank:
                        hs = G + up(X * rb)
                        bank[key] = (ok(_cu().cuMemCreate(hs, _vmm()[1], 0)), hs)
                        bank_bytes += hs
                    stage = (Z,) + bank[key]
                mx = MixedRows(R, rb, D, stage)
                new = mx.tensor(old.dtype, (T,) + tuple(old.shape[1:]))
                for s0 in range(0, R, 16):  # rows past R: page padding, then staging
                    new[s0:min(s0 + 16, R)].copy_(old.index_select(0, src_rows[s0:s0 + 16]))
                _keep.append(mx)
                dev_bytes += mx.dsize; host_bytes += mx.hsize
            else:
                new = torch.zeros((T,) + tuple(old.shape[1:]), dtype=old.dtype, device="cuda")
                new[:R] = old.index_select(0, src_rows)
            _set_rows(grp, new)
            c.rows.append(_rows(new))
            (bigv if isbig else smallv).append(c.rows[-1])
            del old
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        runner.routed_experts.global_num_experts = T  # moe_align_block_size drops ids >= this
        _keep.append(c)
        if bstage:
            staged[idx] = (c, D, Z, cold_n[idx], bigv, smallv)
        _wrap(runner, idx, E, cache=c, stage=(bstage, fpos[idx]) if bstage else None,
              ahead=(ahead, fpos[idx]) if ahead else None)
        rowsets[idx] = (name, c, bigv, smallv)
        shape.append((idx, H, S))
    for idx in sorted(staged):
        bstage.add(*staged[idx])
    if bstage:
        _keep.append((bstage, bank))
    if ahead:
        ahead.start([(idx,) + rowsets[idx] for idx in sorted(rowsets)], model, period)
        _keep.append(ahead)
    if hasattr(torch._C, "_host_emptyCache"):
        torch._C._host_emptyCache()
    if period:
        _keep.append(hstats)
        _report(hstats, period)
    shape.sort()
    log(f"cache: {len(layers)} layers, {n_cold} experts cold ({n_cold / len(layers):.1f} a layer), "
        f"{host_bytes / 2**30:.2f} GiB host pages, {dev_bytes / 2**30:.2f} GiB device, policy {pol}, profile {src}; "
        f"slots {SLOTS}: pinned/slots per layer {[f'{h}/{s}' for _, h, s in shape]}; "
        f"evict {EVICT}{SHIFT if EVICT == 'lfu' else ''}; frozen from {FREEZE_M or 'never'} tokens"
        + (f"; DMA from {DMA_M} tokens: {X} staging rows, {nbuf} buffers, {bank_bytes / 2**30:.2f} GiB" if X else "")
        + (f"; ahead {f'from {AHEAD_MIN} ' if AHEAD_MIN > 1 else ''}below {AHEAD_M} tokens, top {AHEAD_K}"
           f"{f', fused to {AHEAD_FUSE_M}' if AHEAD_FUSE else ''}"
           f"{', DRY (no copies)' if AHEAD_DRY else ''}{', waits timed' if AHEAD_TICK else ''}"
           f"{', side stream' if AHEAD_SIDE else ''}{f', stale below {AHEAD_STALE:g}' if AHEAD_STALE else ''}"
           f"{', without ' + '/'.join(sorted(AHEAD_SKIP)) if AHEAD_SKIP else ''}, "
           f"{ahead.ntens} tensors on the copy engine" if ahead else ""))


# ---------------------------------------------------------------- predict: is the next layer's routing knowable early?

PRED_K = (6, 10)  # ids predicted per token: the router's 6, and a wider net
PRED_SLOTS = ("own", "d1 raw", "d1 scaled", "d2 raw", "d2 scaled")
PRED_FIELDS = ("steps", "tokens", "tok_hit", "u_act", "u_hit", "u_pred", "c_act", "c_hit", "c_pred")
PRED_MAXT = int(os.environ.get("KSTAGE_PREDICT_MAXT", "16384"))


def _install_predict(layers, model, cold_n, counts, device="cuda"):
    """A probe, no staging. At each MoE layer, the next two MoE layers' gates are applied to this
    layer's input x; when those layers run, their routed ids are compared with the prediction.
    raw: x as it is. scaled: x * w_j / w_i, this layer's norm weight swapped for the target's,
    which is the target's input but for what the layers between add to the residual. own: this
    layer's gate on its own x, which must reproduce the router (the replica check).
    Per prediction and K in PRED_K, split decode (<= 64 tokens) / prefill: per-token hits, and over
    the batch's union of ids (what a fetch would cover): all ids, and the cold ones only."""
    E = len(counts[layers[0][0]])
    mods = dict(model.named_modules())
    order = sorted(layers, key=lambda l: l[0])
    L, K = len(order), max(PRED_K)
    gates, norms, cold = [], [], []
    for idx, name, _ in order:
        gate = mods[name.rsplit(".", 1)[0]].gate
        assert gate.e_score_correction_bias is not None, f"{name}: no correction bias"
        gates.append((gate.weight, gate.e_score_correction_bias))
        norms.append(mods[name.rsplit(".", 2)[0]].norm.weight.detach().float())
        perm = sorted(range(E), key=lambda e: (-counts[idx][e], -e))  # as _cache_layout
        m = torch.zeros(E, dtype=torch.bool, device=device)
        m[perm[E - cold_n[idx]:]] = True
        cold.append(m)
    ratio = {(i, i + d): norms[i + d] / norms[i] for i in range(L) for d in (1, 2) if i + d < L}
    shape = (len(PRED_SLOTS), len(PRED_K), 2, len(PRED_FIELDS))
    if device == "cuda":
        host, dev = _stats_buffer(L, *shape)
    else:
        host = dev = torch.zeros((L,) + shape, dtype=torch.int64)
    bufs = [torch.zeros(4, PRED_MAXT, K, dtype=torch.int16, device=device) for _ in order]

    def top(logits, j):  # the router's rule: sigmoid, plus the correction bias, top K
        return (logits.sigmoid() + gates[j][1].float()).topk(K, dim=-1).indices

    def probe(li, x, ids, M):
        xf = x.float()
        n = 1 + 2 * min(li, 2)
        own = top(xf @ gates[li][0].float().t(), li)
        P = torch.cat([own[None].to(torch.int16), bufs[li][:n - 1, :M]])  # [n, M, K]
        act = ids.reshape(M, -1)
        hit = (P[..., None] == act[None, :, None, :]).any(-1)  # [n, M, K]: predicted id is routed
        am = torch.zeros(E, dtype=torch.bool, device=x.device).scatter_(0, act.reshape(-1).long(), True)
        pm = torch.stack([torch.zeros(n, E, dtype=torch.bool, device=x.device)
                          .scatter_(1, P[..., :k].reshape(n, -1).long(), True) for k in PRED_K], 1)
        c = cold[li]
        ac = am & c
        V = torch.stack([torch.stack([hit[..., :k].sum((1, 2)) for k in PRED_K], 1),
                         am.sum().expand(n, len(PRED_K)), (pm & am).sum(-1), pm.sum(-1),
                         ac.sum().expand(n, len(PRED_K)), (pm & ac).sum(-1), (pm & c).sum(-1)], -1)
        row = dev[li, :n, :, 0 if M <= 64 else 1]
        row[..., 0].add_(1)
        row[..., 1].add_(M)
        row[..., 2:].add_(V)
        for d in (1, 2):
            j = li + d
            if j < L:
                w = gates[j][0].float()
                lg = torch.stack([xf @ w.t(), xf @ (w * ratio[(li, j)]).t()])
                bufs[j][2 * d - 2: 2 * d, :M] = top(lg, j).to(torch.int16)

    def wrap(li, re_):
        fwd = re_.forward_modular

        def forward_modular(x, topk_weights, topk_ids, *a, **k):
            M = x.shape[0]
            if 0 < M <= PRED_MAXT:  # the same M reaches every layer of one forward
                probe(li, x, topk_ids, M)
            return fwd(x, topk_weights, topk_ids, *a, **k)

        re_.forward_modular = forward_modular

    for li, (_, _, runner) in enumerate(order):
        wrap(li, runner.routed_experts)
    wmin = min(float(w.abs().min()) for w in norms)
    log(f"predict: {L} layers, {sum(cold_n.values())} cold experts, norm |w| min {wmin:.3g}, "
        f"top {PRED_K}, batches of up to {PRED_MAXT} tokens")
    return host


def _predict_lines(v):
    """v: the stats summed over layers, [slot, K, phase, field]."""
    out = []
    for ph, pname in ((0, "decode"), (1, "prefill")):
        if not int(v[0, 0, ph, 0]):
            continue
        for si, sname in enumerate(PRED_SLOTS):
            parts = []
            for ki, k in enumerate(PRED_K):
                st, tok, th, ua, uh, up, ca, ch, cp = v[si, ki, ph].tolist()
                if st:
                    parts.append(f"top{k} token {th / (tok * 6):.3f}, batch {uh / max(ua, 1):.3f} of "
                                 f"{ua / st:.1f} fetching {up / st:.1f}, cold {ch / max(ca, 1):.3f} of "
                                 f"{ca / st:.2f} fetching {cp / st:.2f}")
            if parts:
                out.append(f"predict {pname} {sname}: {int(v[si, 0, ph, 0])} layer-steps, "
                           f"{int(v[si, 0, ph, 1])} tokens; " + "; ".join(parts))
    return out


def _report_predict(host, period):
    import threading

    def run():
        last = None
        while True:
            time.sleep(period)
            v = host.sum(0)
            if last is None or not torch.equal(v, last):
                for line in _predict_lines(v):
                    log(line)
                d = host[:, 2, 0, 0]  # d1 scaled, top6, decode, per layer
                if int(d[:, 0].sum()):
                    log("predict decode d1 scaled top6 token recall by layer: "
                        f"{[round(int(h) / max(int(t) * 6, 1), 2) for t, h in zip(d[:, 1], d[:, 2])]}")
            last = v.clone()
    threading.Thread(target=run, daemon=True, name="kstage-predict").start()


def _install(model):
    mode = os.environ.get("KSTAGE", "")
    layers = _layers(model)
    if not layers:
        log("no MoE layers found; nothing done")
        return
    t0 = time.perf_counter()
    pred = None
    if mode == "predict" or os.environ.get("KSTAGE_PREDICT") == "1":  # the split before any mode moves rows
        E = layers[0][2].routed_experts.w13_weight.shape[0]
        cold_n, counts, *_ = _cold_split(layers, {idx: _groups(r) for idx, _, r in layers}, E)
    if mode != "predict":
        {"gather": _install_gather, "hotcold": _install_hotcold,
         "cache": lambda l: _install_cache(l, model)}[mode](layers)
    if mode == "predict" or os.environ.get("KSTAGE_PREDICT") == "1":
        pred = _install_predict(layers, model, cold_n, counts)
        _keep.append(pred)
        _report_predict(pred, float(os.environ.get("KSTAGE_STATS", "0")) or 30)
    torch.cuda.synchronize()
    log(f"installed in {time.perf_counter() - t0:.1f}s; device memory free "
        f"{torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB")


def _pf_fix(cls):
    """KSTAGE_PFFIX=1: make vLLM's prefetch offloader (--offload-backend prefetch) safe when
    the number of offloaded modules n is not a multiple of the step s. Module i reads slot
    i % s, and after its forward starts the copy of module (i + s) % n into that module's
    slot. For a wrap-around target j = i + s - n with n % s != 0, slot j % s is still to be
    read later in the same pass (n=7, s=2: module 5 starts module 0's copy into slot 0
    while module 6, also slot 0, has not run), so the copy races the read. Here such a
    copy starts after the slot's last reader in the pass instead."""
    orig = cls._start_prefetch
    if getattr(orig, "_kstage", False):
        return

    def _start_prefetch(self, j):
        n, s = len(self.module_offloaders), self.prefetch_step
        if n % s == 0:
            return orig(self, j)
        i = (j - s) % n  # the module whose forward just ended
        pend = self.__dict__.setdefault("_kstage_pend", {})
        last = max(m for m in range(n) if m % s == j % s)
        if i + s >= n and last > i:
            pend.setdefault(last, []).append(j)
        else:
            orig(self, j)
        for k in pend.pop(i, []):
            orig(self, k)

    _start_prefetch._kstage = True
    cls._start_prefetch = _start_prefetch
    log("prefetch offloader: wrap-around copies wait for their slot's last reader")


_lora_ids = [None]  # the router's own ids while a wrapped layer runs


def _lora_fix(cls):
    """MoE LoRA indexes each expert's adapter by the ids Marlin gets (punica add_lora_w13;
    w2 reuses its alignment). Hotcold permutes those ids and cache sends rows up to E + slots,
    past the adapter's E experts (decode's naive block assignment does not filter them), so
    the LoRA call gets the router's ids instead."""
    orig = cls.apply_w13_lora
    if getattr(orig, "_kstage", False):
        return

    def apply_w13_lora(self, lora_context, **kw):
        ids = _lora_ids[0]
        if ids is not None and kw.get("topk_ids") is not None and kw["topk_ids"].shape == ids.shape:
            kw["topk_ids"] = ids
        return orig(self, lora_context, **kw)

    apply_w13_lora._kstage = True
    cls.apply_w13_lora = apply_w13_lora


def register():
    if os.environ.get("KSTAGE_PFFIX") == "1":
        from vllm.model_executor.offloader import prefetch
        _pf_fix(prefetch.PrefetchOffloader)
    mode = os.environ.get("KSTAGE", "")
    if not mode:
        return
    assert mode in ("gather", "hotcold", "cache", "predict"), \
        f"KSTAGE={mode}: expected gather, hotcold, cache or predict"
    from vllm.model_executor.layers.fused_moe.experts import lora_experts_mixin
    _lora_fix(lora_experts_mixin.LoRAExpertsMixin)
    from vllm.model_executor.model_loader import base_loader
    orig = base_loader.BaseModelLoader.load_model
    if getattr(orig, "_kstage", False):
        return

    def load_model(self, *a, **k):
        model = orig(self, *a, **k)
        _install(model)
        return model

    load_model._kstage = True
    base_loader.BaseModelLoader.load_model = load_model
    log(f"registered, mode {mode}")

"""Shared input builder: reproduces vLLM 0.29 mamba_mixer2 decode-time SSU call
for Nemotron-3.5-Lightning (nemotron_h): 64 heads x head_dim 64, dstate 128,
n_groups 8, ssm state float32, activations bf16.

Layout mirrors mamba_mixer2.conv_ssm_forward / forward decode branch:
  projected_states (T, 4096 gate | 6144 xBC | 64 dt) bf16
  x  = xBC[:, :4096].view(T, 64, 64)         strides (10304, 64, 1)
  B  = xBC[:, 4096:5120].view(T, 8, 128)     strides (10304, 128, 1)
  C  = xBC[:, 5120:6144].view(T, 8, 128)
  dt = dt[:, :, None].expand(-1, -1, 64)     strides (10304, 1, 0)
  A  = A[:, None, None].expand(-1, 64, 128).float()   (stride-0 broadcast)
  D, dt_bias = param[:, None].expand(-1, 64)  bf16
  state_batch_indices = dst = (T, 1) int32 (mamba_cache_mode != "all")
  out = (T, 4096) bf16 viewed (T, 64, 64)
  ssm_state: 'paged' = strided view into a per-slot page that also holds the
  conv state (3 x 6144 bf16 = 36864 B) in front, like vLLM's shared page.
"""
import ctypes
import os
import torch

if os.environ.get("SSU_PATCH_DEDUP") == "1":
    # Source agent's H4 test: build the SSU module without the explicit-
    # instantiation TU, so the .so holds one copy of each kernel. Needs a
    # fresh FLASHINFER_WORKSPACE_BASE (the URI is unchanged).
    import flashinfer.jit.mamba.selective_state_update as _m
    _orig_gen = _m._gen_module

    def _patched_gen(*a, **k):
        spec = _orig_gen(*a, **k)
        spec.sources = [s for s in spec.sources
                        if not str(s).endswith("_kernel_inst.cu")]
        return spec
    _m._gen_module = _patched_gen

NH, HD, DS, NG = 64, 64, 128, 8
GATE = NH * HD                      # 4096
CONV = NH * HD + 2 * NG * DS        # 6144
PROJ = GATE + CONV + NH             # 10304
CONV_STATE_BYTES = 3 * CONV * 2     # 36864
STATE_ELEMS = NH * HD * DS          # 524288 floats = 2 MiB
NULL_BLOCK_ID = 0


def make_state(nslots, layout, device="cuda"):
    if layout == "paged":
        page_f32 = (CONV_STATE_BYTES + STATE_ELEMS * 4) // 4
        raw = torch.zeros(nslots * page_f32, dtype=torch.float32, device=device)
        state = torch.as_strided(raw, (nslots, NH, HD, DS),
                                 (page_f32, HD * DS, DS, 1),
                                 storage_offset=CONV_STATE_BYTES // 4)
    else:
        state = torch.zeros(nslots, NH, HD, DS, dtype=torch.float32, device=device)
    return state


def make_inputs(batch, nslots=None, layout="paged", seed=0, device="cuda"):
    if nslots is None:
        nslots = batch + 1
    g = torch.Generator(device=device).manual_seed(seed)
    proj = torch.randn(batch, PROJ, device=device, dtype=torch.float32,
                       generator=g).to(torch.bfloat16)
    xBC = proj[:, GATE:GATE + CONV]
    dt_raw = proj[:, GATE + CONV:]
    x = xBC[:, :NH * HD].view(-1, NH, HD)
    Bm = xBC[:, NH * HD:NH * HD + NG * DS].view(-1, NG, DS)
    Cm = xBC[:, NH * HD + NG * DS:].view(-1, NG, DS)
    dt = dt_raw[:, :, None].expand(-1, -1, HD)

    A_param = -torch.exp(torch.rand(NH, device=device, generator=g) * 2.0)
    D_param = torch.randn(NH, device=device, generator=g).to(torch.bfloat16)
    dtb_param = (torch.rand(NH, device=device, generator=g) * 2 - 4).to(torch.bfloat16)
    A = A_param[:, None, None].expand(-1, HD, DS).to(dtype=torch.float32)
    D = D_param[:, None].expand(-1, HD)
    dt_bias = dtb_param[:, None].expand(-1, HD)

    state = make_state(nslots, layout, device)
    state.copy_(torch.randn(state.shape, device=device, generator=g) * 0.5)

    perm = torch.randperm(nslots - 1, device="cpu",
                          generator=torch.Generator().manual_seed(seed)) + 1
    idx_buf = torch.zeros(max(batch, 1), 1, dtype=torch.int32, device=device)
    idx_buf[:batch, 0] = perm[:batch].to(torch.int32).to(device)
    idx = idx_buf[:batch]

    out_full = torch.full((batch, GATE), float("nan"), dtype=torch.bfloat16, device=device)
    out = out_full.view(batch, -1, HD)
    return dict(state=state, x=x, dt=dt, A=A, B=Bm, C=Cm, D=D, dt_bias=dt_bias,
                idx=idx, out=out, proj=proj)


def call_flashinfer(inp, algorithm="auto"):
    from flashinfer.mamba import selective_state_update as fi_ssu
    # Same keyword set as vllm ssu_dispatch.FlashInferSSUBackend.__call__
    # (stochastic rounding off -> rand_seed None; cu_seqlens None for non-spec).
    fi_ssu(inp["state"], inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"],
           D=inp["D"], z=None, dt_bias=inp["dt_bias"], dt_softplus=True,
           state_batch_indices=inp["idx"], dst_state_batch_indices=inp["idx"],
           cu_seqlens=None, num_accepted_tokens=None, cache_steps=0,
           pad_slot_id=NULL_BLOCK_ID, out=inp["out"], rand_seed=None,
           philox_rounds=10, algorithm=algorithm)


def call_triton(inp):
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
    selective_state_update(inp["state"], inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"],
                           D=inp["D"], z=None, dt_bias=inp["dt_bias"], dt_softplus=True,
                           state_batch_indices=inp["idx"],
                           dst_state_batch_indices=inp["idx"],
                           null_block_id=NULL_BLOCK_ID, out=inp["out"],
                           num_accepted_tokens=None, cu_seqlens=None,
                           is_blackwell=False, enable_stochastic_rounding=False,
                           cache_philox_rounds=0)


_cudart = None


def cudart_last_error():
    """cudaGetLastError from the libcudart instance already loaded in-process."""
    global _cudart
    if _cudart is None:
        path = None
        with open("/proc/self/maps") as f:
            for line in f:
                if "libcudart.so" in line:
                    path = line.split()[-1]
                    break
        if path is None:
            return None, "no libcudart mapped"
        _cudart = ctypes.CDLL(path)
        _cudart.cudaGetErrorName.restype = ctypes.c_char_p
        _cudart._path = path
    e = _cudart.cudaGetLastError()
    return e, _cudart.cudaGetErrorName(e).decode()

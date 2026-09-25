"""Memory-lean torch path for nemotron_h's mamba2_chunk_scan (G5 option 2).

David approved this variant 2026-09-24, after the G5 desktop debug run OOMed at seq 1024.

With no mamba kernels in the image, transformers 5.16.1 runs mamba2_chunk_scan
(modeling_nemotron_h.py, sha256 d4206742...) in torch. It writes five contractions as
broadcast-then-sum, which builds the full fp32 product before reducing it. At
Lightning's shapes (chunk 128, 64 heads, head dim 64, state 128), per 128-token chunk:
  G           (C * B).sum(-1)            128 x 128 x 64 x 128   0.5  GiB
  Y_diag      (M * x).sum(3)             128 x 128 x 64 x 64    0.25 GiB
  states      (B_decay * x).sum(2)       128 x 64 x 64 x 128    0.25 GiB
  Y_off       (C * states).sum(-1)       128 x 64 x 64 x 128    0.25 GiB
  new_states  (decay_chunk * states).sum(1), quadratic in chunks, 0.16 GiB at seq 1024
The debug run failed on the first ("Tried to allocate 4.00 GiB" = 8 chunks x 0.5 GiB).

`mamba2_chunk_scan` below is the 5.16.1 torch function with only those five lines
rewritten as torch.einsum, which runs each as a batched matmul without building the
product. Everything else, fp32 included, is copied unchanged. `install()` swaps it into
the module; the mixer looks the name up at call time (modeling_nemotron_h.py:542), and
`use_kernelized_func` is a no-op without the `kernels` package.
"""
import hashlib

import torch
import torch.nn.functional as F
import transformers.models.nemotron_h.modeling_nemotron_h as nh

EXPECTED_SHA256 = "d42067429e3268d1163579f5d257fe0baa9a2b02edba5fcb175fb8d42c9b6af4"
ORIGINAL = nh.mamba2_chunk_scan
calls = 0


def mamba2_chunk_scan(
    hidden_states, dt, A, B, C, chunk_size, D=None, dt_bias=None, initial_states=None,
    dt_softplus=False, dt_limit=(0.0, float("inf")), return_final_states=False, **kwargs,
):
    global calls
    calls += 1
    batch_size, sequence_length, num_heads, head_dim = hidden_states.shape
    num_groups = B.shape[2]

    if dt_bias is not None:
        dt = dt + dt_bias.to(dt.dtype)
    if dt_softplus:
        dt = F.softplus(dt)
    dt = torch.clamp(dt, min=dt_limit[0], max=dt_limit[1])

    hidden_states = hidden_states.float()
    B = B.float().repeat_interleave(num_heads // num_groups, dim=2, output_size=num_heads)
    C = C.float().repeat_interleave(num_heads // num_groups, dim=2, output_size=num_heads)

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    D_residual = None
    if D is not None:
        D_residual = D[..., None] * nh.pad_tensor_by_size(hidden_states, pad_size)

    hidden_states = hidden_states * dt[..., None].float()
    A = A.to(hidden_states.dtype) * dt.float()

    hidden_states, A, B, C = [nh.reshape_into_chunks(t, pad_size, chunk_size) for t in (hidden_states, A, B, C)]

    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)

    L = torch.exp(nh.segment_sum(A))

    # was: (C[:, :, :, None, :, :] * B[:, :, None, :, :, :]).sum(dim=-1)
    G = torch.einsum("bclhn,bcshn->bclsh", C, B)

    M = (G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]).sum(dim=-1)

    # was: (M[..., None] * hidden_states[:, :, None]).sum(dim=3)
    Y_diag = torch.einsum("bclsh,bcshp->bclhp", M, hidden_states)

    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
    # was: (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)
    states = torch.einsum("bclhn,bclhp->bchpn", B_decay, hidden_states)

    previous_states = (
        initial_states[:, None].to(dtype=states.dtype, device=states.device)
        if initial_states is not None
        else torch.zeros_like(states[:, :1])
    )
    states = torch.cat([previous_states, states], dim=1)
    decay_chunk = torch.exp(nh.segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))).transpose(1, 3)
    # was: (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
    new_states = torch.einsum("bjih,bjhpn->bihpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    state_decay_out = torch.exp(A_cumsum)
    # was: C_times_states = C[..., None, :] * states[:, :, None, ...]; .sum(-1) below
    C_times_states = torch.einsum("bclhn,bchpn->bclhp", C, states)
    Y_off = C_times_states * state_decay_out.permute(0, 2, 3, 1)[..., None]

    output = Y_diag + Y_off
    output = output.reshape(batch_size, -1, num_heads, head_dim)

    if D_residual is not None:
        output = output + D_residual

    if pad_size > 0:
        output = output[:, :sequence_length]

    if return_final_states:
        return output, final_state

    return output


def modeling_sha256():
    with open(nh.__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def install():
    """Swap the lean scan in. Refuses a transformers build this copy was not taken from."""
    sha = modeling_sha256()
    assert sha == EXPECTED_SHA256, f"modeling_nemotron_h.py is {sha}, not the copied 5.16.1 file"
    nh.mamba2_chunk_scan = mamba2_chunk_scan


def report():
    return {"installed": nh.mamba2_chunk_scan is mamba2_chunk_scan, "calls": calls,
            "modeling_sha256": modeling_sha256(), "torch_matmul_precision": torch.get_float32_matmul_precision()}

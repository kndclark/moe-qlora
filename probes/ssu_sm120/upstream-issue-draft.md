<!-- DRAFT, NOT FILED. David, 2026-10-02: do NOT post or file anything; research only.
     Searched 2026-10-02 (plan.md "L10b"): no FlashInfer/vLLM/SGLang/CCCL/CUTLASS issue for
     this; FlashInfer main 46340689 still has the 13 cluster-form calls. -->

# Mamba selective_state_update: first launch reserves ~1.6 GiB on sm_120 (cluster-space TMA load lowers to a syscall)

**Environment:** RTX 5090 Laptop GPU (sm_120, 82 SMs, 24 GB), driver 595.91.07,
`vllm/vllm-openai:v0.29.0` image (FlashInfer 0.6.18, bundled CCCL 3.3.2), nvcc 13.0.88 and
13.3.73. Seen through vLLM `--mamba-backend flashinfer` serving a NemotronH model.

**Symptom:** the first launch of `selective_state_update_kernel_producer_consumer_vertical`
(also horizontal) fails with `CUDA_ERROR_OUT_OF_MEMORY` from `cuLaunchKernel` unless about
1.7 GiB of device memory is free. When it succeeds, free memory drops by 1,634 MiB and
stays down. The launch result is not checked, so the error surfaces at the next op.

**Cause:** `kernel_selective_state_update_{stp,mtp_vertical,mtp_horizontal}.cuh` (13 call
sites) use `cuda::device::experimental::cp_async_bulk_tensor_4d_global_to_shared`, which
emits the `.shared::cluster` form of
`cp.async.bulk.tensor.4d...global.tile.mbarrier::complete_tx::bytes`. On sm_120 / sm_120a /
sm_120f, ptxas lowers that form to a driver syscall
(`__cuda_syscall_cp_async_bulk_tensor_4d_tile_unicast`) that raises the kernel's
per-thread stack from 1,024 to 14,608 B. The driver reserves the stack for every resident
thread: 13,584 B x 82 SMs x 1,536 threads = 1,631.7 MiB, matching the measured 1,634 MiB.
On sm_90a and sm_100a the same form compiles inline (STACK 0). The `.shared::cta` form
compiles inline on sm_120a too (0 syscall references, STACK 0).

**Fix that works here:** these kernels never launch as clusters, so the destination and
mbarrier are always in the issuing CTA and the `.shared::cta` form is equivalent. A small
inline-asm helper issuing `.shared::cta` (libcu++'s
`cp_async_bulk_tensor.h` also provides this form) in place of the 13 calls:
- removes the syscall (0 references in the built module; vertical REG 29 STACK 0);
- first-launch free-memory drop 0 MiB for vertical / horizontal / auto at batch 16 and 64;
- no NaN, and output error vs an fp64 reference identical to the Triton backend at batch
  1 / 16 / 64, paged and contiguous state;
- end to end under vLLM 0.29: serves; over 3 runs each, eval quality ties Triton
  and single-request decode is ~7% faster in eager mode.

A workaround with no code change: `--mamba-ssu-algorithm simple` (no TMA).

Prior art: NVIDIA/cccl#6708 identifies the same driver allocation for `cp_async_bulk` to
`space_cluster` on sm120 (~14.5 KiB per thread, "by design"); CCCL moved its own algorithms
to `space_shared` in NVIDIA/cccl#6362 and has deprecated the
`cuda::device::experimental::cp_async_bulk_tensor_*_global_to_shared` helpers since 3.2.
The supported replacement, `cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_shared,
cuda::ptx::space_global, ...)`, emits the `.shared::cta` form. sm_121a (DGX Spark) and
sm_120f lower the cluster form to the same syscall (nvcc 13.0.88 / 13.3.73).

Patch and repro harness available on request.

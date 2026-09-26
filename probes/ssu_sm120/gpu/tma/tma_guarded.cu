// Minimal control: a kernel whose only TMA load sits behind a never-true
// branch. If the driver still grows the stack at launch, the reservation is
// triggered by the presence of the syscall, not by FlashInfer.
#include <cuda.h>
#include <cstdint>
__device__ __forceinline__ uint32_t s(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
extern "C" __global__ void no_tma(const CUtensorMap* m, int flag, float* out) {
  __shared__ alignas(128) float buf[16 * 128];
  if (flag == 12345) buf[threadIdx.x] = (float)(uintptr_t)m;
  __syncthreads();
  if (threadIdx.x == 0) out[blockIdx.x] = buf[0] * 0.f + 1.f;
}
extern "C" __global__ void tma2d_guarded(const CUtensorMap* m, int flag, float* out) {
  __shared__ alignas(128) float buf[16 * 128];
  __shared__ alignas(8) uint64_t bar;
  if (flag == 12345) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(s(buf)), "l"(m), "r"(0), "r"(0), "r"(s(&bar)) : "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) out[blockIdx.x] = buf[0] * 0.f + 1.f;
}

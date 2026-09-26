// TMA 2D load lowering on sm_120: does the destination state space or the
// ptxas version decide whether it becomes a driver syscall?
#include <cuda.h>
#include <cstdint>
__device__ __forceinline__ uint32_t s(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
extern "C" __global__ void load_cluster(const __grid_constant__ CUtensorMap m, int x, int y) {
  __shared__ alignas(128) float buf[16 * 128];
  __shared__ alignas(8) uint64_t bar;
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(s(buf)), "l"(&m), "r"(x), "r"(y), "r"(s(&bar)) : "memory");
}
#ifdef CTA
extern "C" __global__ void load_cta(const __grid_constant__ CUtensorMap m, int x, int y) {
  __shared__ alignas(128) float buf[16 * 128];
  __shared__ alignas(8) uint64_t bar;
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(s(buf)), "l"(&m), "r"(x), "r"(y), "r"(s(&bar)) : "memory");
}
#endif

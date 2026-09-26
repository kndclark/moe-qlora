// Which TMA PTX forms does ptxas lower to a driver syscall, per target?
#include <cuda.h>
#include <cstdint>
__device__ __forceinline__ uint32_t s(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
extern "C" __global__ void tma2d(const __grid_constant__ CUtensorMap m, int x, int y) {
  __shared__ alignas(128) float buf[16 * 128];
  __shared__ alignas(8) uint64_t bar;
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(s(buf)), "l"(&m), "r"(x), "r"(y), "r"(s(&bar)) : "memory");
}
extern "C" __global__ void tma4d(const __grid_constant__ CUtensorMap m, int a, int b, int c, int d) {
  __shared__ alignas(128) float buf[16 * 128];
  __shared__ alignas(8) uint64_t bar;
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];"
               :: "r"(s(buf)), "l"(&m), "r"(a), "r"(b), "r"(c), "r"(d), "r"(s(&bar)) : "memory");
}
extern "C" __global__ void tma4d_store(const __grid_constant__ CUtensorMap m, int a, int b, int c, int d) {
  __shared__ alignas(128) float buf[16 * 128];
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];"
               :: "l"(&m), "r"(a), "r"(b), "r"(c), "r"(d), "r"(s(buf)) : "memory");
}

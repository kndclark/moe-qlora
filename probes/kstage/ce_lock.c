// ce_lock (row O): why the KSTAGE_AHEAD helper deadlocks inside vLLM, and which helper placement
// escapes it. gdb on the hung engine (p18) showed the main thread inside cuMemcpyDtoHAsync to
// pageable memory, waiting for a stream that waits (cuStreamWaitValue32) on the helper's flag,
// and the helper blocked on a driver rwlock inside cuMemcpyDtoDAsync. Here stream S1 waits on a
// flag; the helper sleeps 300 ms, copies 64 MiB and writes the flag; meanwhile the main thread
// makes one blocking call:
//   d2h   cuMemcpyDtoHAsync on S1 into pageable memory (what vLLM's step loop did)
//   sync  cuStreamSynchronize(S1)
//   free  cuMemFree of an unrelated buffer
// Helper placements: same (a thread on the main context, as ce_helper.c), ctx (a thread with a
// context of its own), proc (a child process; the buffers are VMM allocations shared as POSIX fd
// handles, the source on host pages as in vllm_kstage's MixedRows, but HOST_NUMA node 0: plain
// HOST pages refuse a POSIX fd handle type, cuMemCreate: invalid argument). If the main call has not
// returned after 10 s it is a deadlock. op overlap: does the helper's copy (16 x 256 MiB, host
// pages to device) run while a kernel on every SM of the main context does?
//   ce_lock <same|ctx|proc> <d2h|sync|free|overlap>
#include <cuda.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

typedef struct { CUdeviceptr va; size_t size; CUmemGenericAllocationHandle h; } vmm_t;

static void map(vmm_t *v) {
  CK(cuMemAddressReserve(&v->va, v->size, 0, 0, 0));
  CK(cuMemMap(v->va, v->size, 0, v->h, 0));
  CUmemAccessDesc a;
  memset(&a, 0, sizeof a);
  a.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CK(cuMemSetAccess(v->va, v->size, &a, 1));
}

static vmm_t vmm(size_t want, int host) {
  vmm_t v = {0};
  CUmemAllocationProp p;
  memset(&p, 0, sizeof p);
  p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  p.location.type = host ? CU_MEM_LOCATION_TYPE_HOST_NUMA : CU_MEM_LOCATION_TYPE_DEVICE;  // plain HOST: no fd handle
  p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  size_t g;
  CK(cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  v.size = (want + g - 1) / g * g;
  CK(cuMemCreate(&v.h, v.size, &p, 0));
  map(&v);
  return v;
}

static int export_fd(vmm_t *v) {
  int fd;
  CK(cuMemExportToShareableHandle(&fd, v->h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  fcntl(fd, F_SETFD, 0);  // survives the child's exec
  return fd;
}

static vmm_t import_fd(int fd, size_t size) {
  vmm_t v = {0};
  v.size = size;
  CK(cuMemImportFromShareableHandle(&v.h, (void *)(uintptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  map(&v);
  return v;
}

typedef struct { int go_r, rep_w; CUdeviceptr flag, dst, src; } hctx_t;

// One task per byte read: 'd' sleep 300 ms, copy 64 MiB, write the flag; 'o' copy 16 x 256 MiB;
// 'q' quit. Reports three doubles: the copy calls returned (s after they began), the task ended
// (s after the byte arrived), the copies took on the GPU (s, events).
static void helper_loop(hctx_t *h) {
  CUstream s;
  CUevent e0, e1;
  CK(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));
  CK(cuEventCreate(&e0, 0));
  CK(cuEventCreate(&e1, 0));
  char ready = 'r';
  if (write(h->rep_w, &ready, 1) != 1) exit(1);
  for (;;) {
    char t;
    if (read(h->go_r, &t, 1) != 1 || t == 'q') break;
    double t0 = now(), r[3];
    if (t == 'd') usleep(300000);
    int reps = t == 'd' ? 1 : 16;
    size_t b = (size_t)(t == 'd' ? 64 : 256) << 20;
    double a = now();
    CK(cuEventRecord(e0, s));
    for (int i = 0; i < reps; i++) CK(cuMemcpyDtoDAsync(h->dst, h->src, b, s));
    CK(cuEventRecord(e1, s));
    if (t == 'd') CK(cuStreamWriteValue32(s, h->flag, 1, CU_STREAM_WRITE_VALUE_DEFAULT));
    r[0] = now() - a;
    CK(cuStreamSynchronize(s));
    float ms;
    CK(cuEventElapsedTime(&ms, e0, e1));
    r[1] = now() - t0;
    r[2] = ms / 1e3;
    if (write(h->rep_w, r, sizeof r) != sizeof r) exit(1);
  }
}

typedef struct { CUcontext ctx; CUdevice dev; int own; hctx_t h; } targ_t;

static void *helper_thread(void *p) {
  targ_t *a = (targ_t *)p;
  CUcontext c = a->ctx;
  if (a->own) CK(cuCtxCreate(&c, NULL, 0, a->dev));
  CK(cuCtxSetCurrent(c));
  helper_loop(&a->h);
  return NULL;
}

static int child(char **argv) {  // child <flag fd> <size> <dst fd> <size> <src fd> <size> <go_r> <rep_w>
  CUdevice dev;
  CUcontext ctx;
  CK(cuInit(0));
  CK(cuDeviceGet(&dev, 0));
  CK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CK(cuCtxSetCurrent(ctx));
  hctx_t h;
  h.flag = import_fd(atoi(argv[2]), strtoull(argv[3], 0, 10)).va;
  h.dst = import_fd(atoi(argv[4]), strtoull(argv[5], 0, 10)).va;
  h.src = import_fd(atoi(argv[6]), strtoull(argv[7], 0, 10)).va;
  h.go_r = atoi(argv[8]);
  h.rep_w = atoi(argv[9]);
  helper_loop(&h);
  return 0;
}

static pid_t kid;
static void on_alarm(int sig) {
  (void)sig;
  static const char m[] = "DEADLOCK: the main call has not returned after 10 s\n";
  if (write(1, m, sizeof m - 1)) {}
  if (kid) kill(kid, SIGKILL);
  _exit(2);
}

static const char *ptx =
  ".version 7.0\n.target sm_52\n.address_size 64\n"
  ".visible .entry spin(.param .u64 n, .param .u64 out)\n{\n"
  "  .reg .pred %p;\n  .reg .b64 %i, %n, %x, %o;\n"
  "  ld.param.u64 %n, [n];\n  ld.param.u64 %o, [out];\n  mov.u64 %i, 0;\n  mov.u64 %x, 1;\n"
  "L:\n  mad.lo.u64 %x, %x, 6364136223846793005, 1442695040888963407;\n"
  "  add.u64 %i, %i, 1;\n  setp.lt.u64 %p, %i, %n;\n  @%p bra L;\n"
  "  setp.eq.u64 %p, %x, 0;\n  @%p st.global.u64 [%o], %x;\n  ret;\n}\n";

int main(int argc, char **argv) {
  if (argc > 1 && !strcmp(argv[1], "child")) return child(argv);
  const char *mode = argc > 1 ? argv[1] : "same", *op = argc > 2 ? argv[2] : "d2h";
  setvbuf(stdout, NULL, _IOLBF, 0);
  CUdevice dev;
  CUcontext ctx;
  CK(cuInit(0));
  CK(cuDeviceGet(&dev, 0));
  CK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CK(cuCtxSetCurrent(ctx));
  vmm_t flag = vmm(4, 0), dst = vmm(256 << 20, 0), src = vmm(256 << 20, 1);
  CK(cuMemsetD32(flag.va, 0, 1));
  CK(cuMemsetD8(src.va, 7, src.size));
  CK(cuCtxSynchronize());
  int go[2], rep[2];
  if (pipe(go) || pipe(rep)) return 1;
  pthread_t th;
  targ_t ta = {ctx, dev, !strcmp(mode, "ctx"), {go[0], rep[1], flag.va, dst.va, src.va}};
  if (!strcmp(mode, "proc")) {
    char a[9][32];
    snprintf(a[0], 32, "%d", export_fd(&flag)); snprintf(a[1], 32, "%zu", flag.size);
    snprintf(a[2], 32, "%d", export_fd(&dst)); snprintf(a[3], 32, "%zu", dst.size);
    snprintf(a[4], 32, "%d", export_fd(&src)); snprintf(a[5], 32, "%zu", src.size);
    snprintf(a[6], 32, "%d", go[0]); snprintf(a[7], 32, "%d", rep[1]);
    kid = fork();
    if (!kid) {
      execl("/proc/self/exe", argv[0], "child", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], (char *)NULL);
      _exit(127);
    }
  } else {
    pthread_create(&th, NULL, helper_thread, &ta);
  }
  char c;
  if (read(rep[0], &c, 1) != 1) { printf("%s: the helper did not start\n", mode); return 1; }
  CUstream s1;
  CK(cuStreamCreate(&s1, CU_STREAM_NON_BLOCKING));
  double r[3];
  char t;
  if (strcmp(op, "overlap")) {
    void *pg = malloc(4 << 20);
    CUdeviceptr junk = 0;
    if (!strcmp(op, "free")) CK(cuMemAlloc(&junk, 64 << 20));
    CK(cuStreamWaitValue32(s1, flag.va, 1, CU_STREAM_WAIT_VALUE_EQ));
    signal(SIGALRM, on_alarm);
    alarm(10);
    t = 'd';
    if (write(go[1], &t, 1) != 1) return 1;
    double t0 = now();
    if (!strcmp(op, "d2h")) CK(cuMemcpyDtoHAsync(pg, dst.va, 4 << 20, s1));
    else if (!strcmp(op, "sync")) CK(cuStreamSynchronize(s1));
    else CK(cuMemFree(junk));
    double tm = now() - t0;
    if (read(rep[0], r, sizeof r) != sizeof r) return 1;
    CK(cuStreamSynchronize(s1));
    alarm(0);
    printf("%s %s: main call returned after %.0f ms; helper copy calls returned in %.2f ms, "
           "flag written %.0f ms after go, copy %.1f ms on the GPU\n", mode, op, 1e3 * tm, 1e3 * r[0],
           1e3 * r[1], 1e3 * r[2]);
  } else {
    CUmodule m;
    CUfunction f;
    CUdeviceptr out;
    CUevent k0, k1;
    int nsm;
    CK(cuModuleLoadData(&m, ptx));
    CK(cuModuleGetFunction(&f, m, "spin"));
    CK(cuMemAlloc(&out, 8));
    CK(cuEventCreate(&k0, 0));
    CK(cuEventCreate(&k1, 0));
    CK(cuDeviceGetAttribute(&nsm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    uint64_t n = 1 << 16;
    float ms = 0;
    for (int i = 0; i < 3; i++) {  // calibrate: the kernel alone takes ~200 ms
      void *pa[] = {&n, &out};
      CK(cuEventRecord(k0, s1));
      CK(cuLaunchKernel(f, 8 * nsm, 1, 1, 128, 1, 1, 0, s1, pa, NULL));
      CK(cuEventRecord(k1, s1));
      CK(cuStreamSynchronize(s1));
      CK(cuEventElapsedTime(&ms, k0, k1));
      if (i < 2) n = (uint64_t)(n * 200.0 / ms);
    }
    float k_alone = ms;
    t = 'o';
    if (write(go[1], &t, 1) != 1 || read(rep[0], r, sizeof r) != sizeof r) return 1;
    double c_alone = r[2];
    void *pa[] = {&n, &out};
    CK(cuEventRecord(k0, s1));
    CK(cuLaunchKernel(f, 8 * nsm, 1, 1, 128, 1, 1, 0, s1, pa, NULL));
    CK(cuEventRecord(k1, s1));
    if (write(go[1], &t, 1) != 1) return 1;
    CK(cuStreamSynchronize(s1));
    CK(cuEventElapsedTime(&ms, k0, k1));
    if (read(rep[0], r, sizeof r) != sizeof r) return 1;
    printf("%s overlap: kernel (%d SMs x 8 blocks) alone %.1f ms, with the copy %.1f ms; "
           "copy of 4 GiB alone %.1f ms (%.1f GB/s), with the kernel %.1f ms (%.1f GB/s)\n", mode, nsm, k_alone,
           ms, 1e3 * c_alone, 4.295 / c_alone, 1e3 * r[2], 4.295 / r[2]);
  }
  t = 'q';
  if (write(go[1], &t, 1) != 1) return 1;
  if (kid) waitpid(kid, NULL, 0);
  else pthread_join(th, NULL);
  return 0;
}

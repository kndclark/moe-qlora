// Copy-engine helper (row O): a thread that turns copy plans the GPU writes into pinned memory
// into host-to-device copies on its own stream, so a CUDA graph can fill expert slots ahead
// without SMs. Per channel c (one per MoE layer):
//   plan[c] = [n, src0, dst0, src1, dst1, ...]  int64, written by a kernel (rows, not addresses)
//   req[c]  int32 in pinned memory: the graph writes 1 (cuStreamWriteValue32, fenced) after the plan
//   done[c] int32 in device memory: this thread writes 1 after the copies; the graph waits for 1
//           (cuStreamWaitValue32) before the layer that reads the slots, then writes 0.
// The thread makes no CUDA call until a request arrives, and none arrives while graphs are captured.
// CE_WATCH=s: a watchdog thread; if the helper sits inside one CUDA call for s seconds it names the
// call, raises SIGUSR1 (the probe registers faulthandler on it) and exits, so a hang ends itself.
#include <cuda.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <immintrin.h>

typedef struct {
  CUcontext ctx;
  int nchan, ntens, maxk;
  int64_t *plan;             // nchan x (1 + 2 maxk)
  volatile int32_t *req;     // nchan
  CUdeviceptr done;          // nchan x int32
  uint64_t *host_base, *dev_base, *row_bytes;  // nchan x ntens
  volatile int stop, ready;
  int64_t requests, rows, copies;  // counters for the caller
  pthread_t th, wd;
  CUstream s;
  int err, prio;
  volatile int where;        // the CUDA call the helper is inside (0: none)
  volatile double since;     // when it entered it
  double watch;
} ce_t;

static const char *where_name[] = {"no call", "cuMemcpyHtoDAsync", "cuStreamWriteValue32"};

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

static inline void enter(ce_t *c, int w) { c->since = now(); __sync_synchronize(); c->where = w; }

static void *run(void *p) {
  ce_t *c = (ce_t *)p;
  if (cuCtxSetCurrent(c->ctx) != CUDA_SUCCESS) { c->err = 1; c->ready = 1; return NULL; }
  // CE_PRIO=1: the highest stream priority (the probe's question is whether the copies queue
  // behind the graph's cuStreamWaitValue32 on a shared hardware queue).
  int lo = 0, hi = 0;
  const char *pr = getenv("CE_PRIO");
  if (pr && *pr == '1') cuCtxGetStreamPriorityRange(&lo, &hi);
  if (cuStreamCreateWithPriority(&c->s, CU_STREAM_NON_BLOCKING, hi) != CUDA_SUCCESS) { c->err = 2; c->ready = 1; return NULL; }
  c->prio = hi;
  c->ready = 1;
  const int w = 1 + 2 * c->maxk;
  while (!c->stop) {
    int hit = 0;
    for (int ch = 0; ch < c->nchan; ch++) {
      if (c->req[ch] != 1) continue;
      hit = 1;
      c->req[ch] = 0;
      __sync_synchronize();
      volatile int64_t *pl = c->plan + (int64_t)ch * w;
      int64_t n = pl[0];
      if (n > c->maxk) n = c->maxk;
      for (int64_t i = 0; i < n; i++) {
        uint64_t src = (uint64_t)pl[1 + 2 * i], dst = (uint64_t)pl[2 + 2 * i];
        for (int t = 0; t < c->ntens; t++) {
          int k = ch * c->ntens + t;
          uint64_t rb = c->row_bytes[k];
          enter(c, 1);
          if (cuMemcpyHtoDAsync(c->dev_base[k] + dst * rb, (void *)(c->host_base[k] + src * rb), rb, c->s)
              != CUDA_SUCCESS) c->err = 3;
          c->where = 0;
          c->copies++;
        }
      }
      enter(c, 2);
      if (cuStreamWriteValue32(c->s, c->done + 4 * ch, 1, CU_STREAM_WRITE_VALUE_DEFAULT) != CUDA_SUCCESS)
        c->err = 4;
      c->where = 0;
      c->requests++;
      c->rows += n;
    }
    if (!hit) _mm_pause();
  }
  return NULL;
}

static void *watch(void *p) {
  ce_t *c = (ce_t *)p;
  while (!c->stop) {
    int w = c->where;
    double t = now() - c->since;
    if (w && t > c->watch) {
      fprintf(stderr, "CE_WATCH: the helper has been inside %s for %.0f s; requests %ld, rows %ld, copies issued %ld, "
              "err %d\n", where_name[w], t, (long)c->requests, (long)c->rows, (long)c->copies, c->err);
      fflush(stderr);
      kill(getpid(), SIGUSR1);
      sleep(2);
      _exit(3);
    }
    usleep(100000);
  }
  return NULL;
}

void *ce_start(CUcontext ctx, int nchan, int ntens, int maxk, int64_t *plan, int32_t *req, CUdeviceptr done,
               uint64_t *host_base, uint64_t *dev_base, uint64_t *row_bytes) {
  ce_t *c = calloc(1, sizeof(ce_t));
  c->ctx = ctx; c->nchan = nchan; c->ntens = ntens; c->maxk = maxk;
  c->plan = plan; c->req = req; c->done = done;
  size_t m = (size_t)nchan * ntens * sizeof(uint64_t);
  c->host_base = malloc(m); c->dev_base = malloc(m); c->row_bytes = malloc(m);
  for (size_t i = 0; i < (size_t)nchan * ntens; i++) {
    c->host_base[i] = host_base[i]; c->dev_base[i] = dev_base[i]; c->row_bytes[i] = row_bytes[i];
  }
  const char *wt = getenv("CE_WATCH");
  c->watch = wt ? atof(wt) : 0;
  pthread_create(&c->th, NULL, run, c);
  if (c->watch > 0) pthread_create(&c->wd, NULL, watch, c);
  while (!c->ready) usleep(100);  // its stream exists before any capture
  return c;
}

void ce_stats(void *p, int64_t *out) {  // requests, rows, err, stream state, priority, copies, call inside
  ce_t *c = (ce_t *)p;
  out[0] = c->requests; out[1] = c->rows; out[2] = c->err;
  out[3] = c->s ? (int64_t)cuStreamQuery(c->s) : -1;  // 0 idle, 600 work still queued
  out[4] = c->prio; out[5] = c->copies; out[6] = c->where;
}

void ce_stop(void *p) {
  ce_t *c = (ce_t *)p;
  c->stop = 1;
  pthread_join(c->th, NULL);
  if (c->watch > 0) pthread_join(c->wd, NULL);
}

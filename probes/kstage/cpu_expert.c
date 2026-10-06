// Row P: routed experts computed on the CPU, NVFP4 as the checkpoint stores it (no Marlin repack).
// Lightning's expert: y = down(relu(up x)^2), up [I, H], down [H, I]. Each weight row is two e2m1
// values a byte (low nibble first), an e4m3 scale per 16 inputs and one fp32 scale per tensor:
// w = e2m1 * e4m3 * g. AVX2 + FMA (the 275HX has no AVX-512): nibbles -> int8 at twice the value
// (vpshufb) -> fp32, times the block scale, FMA into one accumulator pair per token.
// A pool of spinning threads takes rows in chunks from a shared counter (P- and E-cores differ).
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct { const uint8_t *uw, *us, *dw, *ds; float ug, dg; } expert_t;

static float E4M3[256];

static void init_e4m3(void) {
  for (int i = 0; i < 256; i++) {
    int e = (i >> 3) & 15, m = i & 7;
    float v = e == 0 ? ldexpf(m / 8.f, -6) : (e == 15 && m == 7) ? NAN : ldexpf(1 + m / 8.f, e - 7);
    E4M3[i] = (i & 128) ? -v : v;
  }
}

static inline float hsum(__m256 v) {
  __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  a = _mm_hadd_ps(a, a);
  a = _mm_hadd_ps(a, a);
  return _mm_cvtss_f32(a);
}

// out[t] = row . x[t] for NT tokens; one dequant of the row serves all of them.
static inline __attribute__((always_inline)) void dot(const uint8_t *w, const uint8_t *s, float g, int K,
                                                      const float *const *x, float *out, const int NT) {
  const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
  const __m128i m4 = _mm_set1_epi8(0x0F);
  __m256 a[4][2];
  for (int t = 0; t < NT; t++) a[t][0] = a[t][1] = _mm256_setzero_ps();
  for (int b = 0; b < K / 16; b++) {
    __m128i v = _mm_loadl_epi64((const __m128i *)(w + 8 * b));
    __m128i lo = _mm_and_si128(v, m4), hi = _mm_and_si128(_mm_srli_epi16(v, 4), m4);
    __m128i q = _mm_shuffle_epi8(lut, _mm_unpacklo_epi8(lo, hi));
    __m256 sc = _mm256_set1_ps(E4M3[s[b]]);
    __m256 w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)), sc);
    __m256 w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q, 8))), sc);
    for (int t = 0; t < NT; t++) {
      a[t][0] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x[t] + 16 * b), a[t][0]);
      a[t][1] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x[t] + 16 * b + 8), a[t][1]);
    }
  }
  for (int t = 0; t < NT; t++) out[t] = hsum(_mm256_add_ps(a[t][0], a[t][1])) * g * 0.5f;
}

static void dot_n(const uint8_t *w, const uint8_t *s, float g, int K, const float *const *x, float *out, int n) {
  for (; n > 0; n -= 4, x += 4, out += 4) switch (n < 4 ? n : 4) {
      case 1: dot(w, s, g, K, x, out, 1); break;
      case 2: dot(w, s, g, K, x, out, 2); break;
      case 3: dot(w, s, g, K, x, out, 3); break;
      default: dot(w, s, g, K, x, out, 4);
    }
}

// ---------------------------------------------------------------- pool

typedef struct pool {
  int n;
  pthread_t *th;
  int gen, stop, pending, bar_cnt, bar_gen;
  void (*fn)(struct pool *, int);
  void *arg;
  int64_t next[2];
} pool_t;

static void barrier(pool_t *p) {
  int g = __atomic_load_n(&p->bar_gen, __ATOMIC_ACQUIRE);
  if (__atomic_add_fetch(&p->bar_cnt, 1, __ATOMIC_ACQ_REL) == p->n) {
    __atomic_store_n(&p->bar_cnt, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&p->bar_gen, g + 1, __ATOMIC_RELEASE);
  } else
    while (__atomic_load_n(&p->bar_gen, __ATOMIC_ACQUIRE) == g) _mm_pause();
}

typedef struct { pool_t *p; int tid; } targ_t;

static void *worker(void *a) {
  pool_t *p = ((targ_t *)a)->p;
  int tid = ((targ_t *)a)->tid, seen = 0;
  free(a);
  for (;;) {
    int g;
    while ((g = __atomic_load_n(&p->gen, __ATOMIC_ACQUIRE)) == seen) _mm_pause();
    seen = g;
    if (__atomic_load_n(&p->stop, __ATOMIC_ACQUIRE)) return NULL;
    p->fn(p, tid);
    __atomic_sub_fetch(&p->pending, 1, __ATOMIC_ACQ_REL);
  }
}

void *ex_pool(int n) {
  init_e4m3();
  pool_t *p = calloc(1, sizeof(pool_t));
  p->n = n;
  p->th = calloc(n, sizeof(pthread_t));
  for (int i = 1; i < n; i++) {
    targ_t *a = malloc(sizeof(targ_t));
    a->p = p; a->tid = i;
    pthread_create(&p->th[i], NULL, worker, a);
  }
  return p;
}

static void run(pool_t *p, void (*fn)(pool_t *, int), void *arg) {
  p->fn = fn; p->arg = arg;
  p->next[0] = p->next[1] = 0;
  __atomic_store_n(&p->pending, p->n - 1, __ATOMIC_RELEASE);
  __atomic_add_fetch(&p->gen, 1, __ATOMIC_ACQ_REL);
  fn(p, 0);
  while (__atomic_load_n(&p->pending, __ATOMIC_ACQUIRE)) _mm_pause();
}

void ex_pool_free(void *v) {
  pool_t *p = v;
  __atomic_store_n(&p->stop, 1, __ATOMIC_RELEASE);
  __atomic_add_fetch(&p->gen, 1, __ATOMIC_ACQ_REL);
  for (int i = 1; i < p->n; i++) pthread_join(p->th[i], NULL);
  free(p->th); free(p);
}

// ---------------------------------------------------------------- MoE on the CPU

enum { CH = 16 };  // rows a grab

typedef struct {
  int H, I, ne;
  const expert_t *ex;
  const int *off, *tok;  // expert e serves tok[off[e]:off[e+1]]
  const float *tw, *x;   // routing weight per pair; x [tokens, H]
  float *y, *h;          // y [tokens, H] (accumulated into); h [pairs, I]
} moe_t;

static void moe_fn(pool_t *p, int tid) {
  moe_t *m = p->arg;
  const int H = m->H, I = m->I;
  const float *xs[64];
  float out[64];
  (void)tid;
  // phase 1: h[pair, i] = relu(up_e[i] . x[tok])^2, rows of every expert in one counter
  for (int64_t r0; (r0 = __atomic_fetch_add(&p->next[0], CH, __ATOMIC_RELAXED)) < (int64_t)m->ne * I;) {
    for (int64_t r = r0; r < r0 + CH && r < (int64_t)m->ne * I; r++) {
      int e = r / I, i = r % I, n = m->off[e + 1] - m->off[e];
      const expert_t *E = &m->ex[e];
      for (int j = 0; j < n; j++) xs[j] = m->x + (int64_t)m->tok[m->off[e] + j] * H;
      dot_n(E->uw + (int64_t)i * (H / 2), E->us + (int64_t)i * (H / 16), E->ug, H, xs, out, n);
      for (int j = 0; j < n; j++) {
        float v = out[j] > 0 ? out[j] : 0;
        m->h[(int64_t)(m->off[e] + j) * I + i] = v * v;
      }
    }
  }
  barrier(p);
  // phase 2: y[tok, o] += w * down_e[o] . h[pair]; a thread owns its rows o across every expert
  for (int64_t o0; (o0 = __atomic_fetch_add(&p->next[1], CH, __ATOMIC_RELAXED)) < H;) {
    for (int o = o0; o < o0 + CH && o < H; o++)
      for (int e = 0; e < m->ne; e++) {
        int n = m->off[e + 1] - m->off[e];
        const expert_t *E = &m->ex[e];
        for (int j = 0; j < n; j++) xs[j] = m->h + (int64_t)(m->off[e] + j) * I;
        dot_n(E->dw + (int64_t)o * (I / 2), E->ds + (int64_t)o * (I / 16), E->dg, I, xs, out, n);
        for (int j = 0; j < n; j++) m->y[(int64_t)m->tok[m->off[e] + j] * H + o] += m->tw[m->off[e] + j] * out[j];
      }
  }
}

// Returns 0, or -1 if an expert serves more than 64 tokens.
int ex_moe(void *pool, int H, int I, int ne, const expert_t *ex, const int *off, const int *tok, const float *tw,
           const float *x, float *y, float *h) {
  for (int e = 0; e < ne; e++)
    if (off[e + 1] - off[e] > 64) return -1;
  moe_t m = {H, I, ne, ex, off, tok, tw, x, y, h};
  run(pool, moe_fn, &m);
  return 0;
}

// ---------------------------------------------------------------- RAM read rate

typedef struct { const uint8_t *p; int64_t n; double sum[64]; } rd_t;

static void rd_fn(pool_t *p, int tid) {
  rd_t *r = p->arg;
  const int64_t C = 1 << 20;
  __m256i acc = _mm256_setzero_si256();
  for (int64_t c0; (c0 = __atomic_fetch_add(&p->next[0], C, __ATOMIC_RELAXED)) < r->n;) {
    int64_t c1 = c0 + C < r->n ? c0 + C : r->n;
    for (int64_t i = c0; i + 128 <= c1; i += 128) {
      __m256i a = _mm256_load_si256((const __m256i *)(r->p + i)), b = _mm256_load_si256((const __m256i *)(r->p + i + 32));
      __m256i c = _mm256_load_si256((const __m256i *)(r->p + i + 64)), d = _mm256_load_si256((const __m256i *)(r->p + i + 96));
      acc = _mm256_add_epi64(acc, _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_xor_si256(c, d)));
    }
  }
  int64_t v[4];
  _mm256_storeu_si256((__m256i *)v, acc);
  r->sum[tid & 63] = (double)(v[0] ^ v[1] ^ v[2] ^ v[3]);
}

// Reads n bytes (32-byte aligned) with every thread; returns a checksum so nothing is elided.
double ex_read(void *pool, const uint8_t *p, int64_t n) {
  rd_t r = {p, n, {0}};
  run(pool, rd_fn, &r);
  double s = 0;
  for (int i = 0; i < 64; i++) s += r.sum[i];
  return s;
}

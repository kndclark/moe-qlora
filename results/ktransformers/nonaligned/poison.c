// LD_PRELOAD: fill posix_memalign blocks >= POISON_MIN bytes with POISON_BYTE, log each to stderr.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int posix_memalign(void **p, size_t a, size_t n) {
  static int (*real)(void **, size_t, size_t);
  if (!real) real = dlsym(RTLD_NEXT, "posix_memalign");
  int rc = real(p, a, n);
  const char *b = getenv("POISON_BYTE"), *m = getenv("POISON_MIN");
  if (!rc && b && n >= (size_t)(m ? atol(m) : 65536)) {
    memset(*p, (int)strtol(b, 0, 0), n);
    fprintf(stderr, "[poison] %zu bytes\n", n);
  }
  return rc;
}

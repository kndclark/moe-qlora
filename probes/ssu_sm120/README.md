# FlashInfer Mamba SSU on sm_120

Why vLLM 0.29's `--mamba-backend flashinfer` fails on the RTX 5090 Laptop, and a local
header patch that fixes it. Findings and decisions: `docs/plan.md`, "Side quest:
FlashInfer SSU". Logs: `results/ssu_sm120/`.

- `patch/include/` — the three FlashInfer 0.6.18 SSU headers with the `.shared::cta` TMA
  shim (`patch/shim.txt`); `patch/orig/` — the pristine copies from the image.
  `diff -u patch/orig patch/include` is the whole patch.
- `serve/` — serve-level A/B. `run_all.sh RUN [LABEL...]` serves each config through
  `serve_ab.sh` (G7a flags) and runs bench.py c=1/c=16 plus the v1 eval, thinking off;
  `compare.py` tabulates every run and applies plan.md's CHOSEN rule.
- `gpu/` — kernel-level harness (reproduces vLLM's SSU call): `probe_launch.py`
  (first-launch memory drop), `sweep_*.py` (free-memory bisect), `bench*.py` (speed),
  `tma/` (control kernels). Run through `gpu/run.sh NAME script.py ...` (unpatched) or
  `patch/run_cta.sh NAME script.py ...` (patched). JIT caches go to
  `$SSU_CACHE` (default `~/.cache/ssu_sm120`).
- `tma2/` — CPU-only compile test: `.shared::cluster` vs `.shared::cta` on sm_120a vs
  sm_100a (syscall refs, REG, STACK).

The kernel harness and the tma2 test ran from a scratch directory; only their paths were
changed when they moved here, and they have not been re-run from this location.

# Plan: 4-bit LoRA of Nemotron 3.5 Lightning on one 24 GB GPU

Written 2026-09-24, after Phase 0 (`docs/evidence.md`). The tags follow the same rules as
there: **MEASURED**, **SOURCED**, **ARITHMETIC**, **UNKNOWN** (plus the experiment that
settles it). This file adds one more: **CHOSEN**, for a threshold picked here.
Each CHOSEN threshold is fixed before its gate runs and is never moved after the result
comes in. David can change any of them *before* the run.

## Question and decision rule

Can Lightning (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, rev `a9904d24`) be
LoRA-trained on one 24 GB card by holding its routed experts in 4-bit, well enough to
replace Qwen3-8B as the lab's default adapter base?

- **Feasible** = G1, G2, G4 and G5 all pass on the laptop (RTX 5090 Laptop, sm_120).
  The desktop 3090 (sm_86) runs G1 and G4 too. A desktop failure is recorded but does not
  block, because the first target is one card. If G5 fails on memory on the laptop, the
  verdict is "not feasible on one card", and the next experiment is a layer split across
  both cards (the 48 GB pool). Cross-node training is proven by
  `~/gpu-lab/bench/pipeline_poc.py` but not built. (Amended 2026-09-24 by David, before
  G5 ran; no CHOSEN threshold changed.)
- **Worth it** = feasible, *and* G6 shows the adapter beating base Lightning on the lab's
  held-out eval. The lab's own evals have already had base + catalog beat tuned adapters
  (see "Why it might not matter"), so feasibility alone does not justify switching.

## Routes

**Route 1 (first): the built-in class, with the experts quantized as they load.**
- transformers 5.16.1's own `nemotron_h` class, with `BitsAndBytesConfig` NF4 for every
  `nn.Linear`.
- Each fused `(128, out, in)` expert Parameter is passed to bnb 0.50.2
  `replace_parameter_4bit` the moment it reaches the GPU. The hook is
  `probes/g4_route1_load.py`, adapted from axolotl 0.19.0 `patch_moe_quantization_on_load`
  (SOURCED raw).
- Why first:
  - It is exact at toy scale in this image (MEASURED, H3).
  - It keeps the `grouped_mm` expert path.
  - It needs no new dependency.
- Known costs:
  - Every access dequantizes one layer's whole expert stack, 2.38 GiB in bf16.
  - No mamba kernels are installed, so mamba runs its torch path.
    `mamba_ssm` and `causal_conv1d` both fail to import in the image (MEASURED, G4 smoke
    run).

**Route 2 (comparison or fallback): NVIDIA remote code, standard bnb.**
- Super's `modeling_nemotron_h.py` (rev `2dc98e2a`, sha256 in evidence.md) with a stock
  `BitsAndBytesConfig`. It quantizes all 29.37B expert params as per-expert `Linear4bit`
  (MEASURED on meta, H1).
- Triggered only if Route 1 fails G4 or G5, or if placement B (below) is wanted and
  Route 1's PEFT patch fails.
- Needs, before it can run:
  1. `mamba_ssm` + `causal_conv1d` built for sm_120 and sm_86. That is a new dependency
     and an image change, so it needs David's OK.
  2. The two unsloth-zoo #1340 fixes: `y*y` in relu2, and a cast of the uint8 dummy input.
- Its MoE forward is a Python loop over all 128 experts with dummy work on idle ones, so
  its speed is UNKNOWN until G5 is re-run on it.

**LoRA placement.**
- **A (default):**
  - attention `q/k/v/o_proj` (6 layers)
  - mamba `in_proj` only (23)
  - `shared_experts.up_proj/down_proj` (23)

  PEFT 0.21 refuses mamba `out_proj` and `conv1d` on nemotron_h (MEASURED).
- **B (optional, later):** add routed-expert LoRA.
  - Route 1 needs axolotl's `patch_peft_target_parameters_matching`, which was written for
    PEFT 0.20; the image has 0.21. Whether it works is UNKNOWN; the h6 probe with the
    patch applied settles it.
  - Route 2 gets standard per-expert targets.

## Storage arithmetic (resident weights only; this is not a peak)

- Routed experts: 23 × 128 × 2 × 2688 × 1856 = 29,374,808,064 params.
  - bf16: 58.75 GB = 54.7 GiB.
  - NF4 packed: 0.5 B/param = 14.69 GB.
  - Double-quantized statistics, bnb 4-bit blocks of 64 values: one uint8 absmax per block
    (29.37e9 / 64 = 459.0M B), plus one fp32 per 256 blocks (7.2M B) = 0.466 GB.
- Other `Linear4bit`: 30.86B − 29.37B = 1.49B params → 0.745 GB packed + ~0.024 GB
  statistics. (Counts MEASURED on meta with NVIDIA's classes. The built-in class may
  differ slightly; G4 counts it.)
- Everything else stays bf16: 0.71B params × 2 B = 1.42 GB (embedding 352M, `lm_head`
  352M, router, conv1d, norms).
- **Total: ~17.34 GB = 16.15 GiB.** On the laptop's 23.89 GiB card that leaves ~7.7 GiB,
  before other apps (~0.9 GiB today) and the CUDA context.
- Per-layer bf16 transient when Route 1 dequantizes one MoE layer's experts:
  128 × 2 × 2688 × 1856 × 2 B = 2.55 GB = 2.38 GiB.
- Load-time transient: the loader holds 128 per-expert bf16 tensors and the merged
  stack at once, 2 × 1.28 GB = 2.55 GB for each Parameter (the loader source was read;
  see the probe docstring). G4 measured a torch peak of 17.564 GiB during load.

## Gates

Order: **G0 → G1 → G4 → G2 → G5 → G7a → G6 → G7**. G7a can move earlier (see below).
Every gate writes `results/<gate>-<label>.json` from a probe in `probes/`.

### G0: download (authorised 2026-09-24) — PASS
- Pinned rev `a9904d24`: 30 files, 65,845,713,051 B (61.32 GiB), 14 safetensors shards.
  The manifest with LFS sha256 values is `results/g0-manifest.json` (SOURCED raw, HF tree
  API).
- The download runs on the desktop straight onto its local disk
  (`/srv/model-cache/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`), as
  uid 1000. That keeps the files readable over NFS; see the root-owned 0600 adapter
  incident in memory.
- **Pass:** every file is present at its manifest size, and every LFS file's sha256
  equals the manifest's, computed on the desktop (local disk).
- **Result (MEASURED, 2026-09-24): PASS.** 30/30 files, 65,845,713,051 B, every sha256
  matches (`results/g0-verify.json`). There is no `refs/` dir; an offline `from_pretrained`
  by the full commit hash works.

### G1: NF4 on real expert weights (sm_120 and sm_86)
- **Method:** stream each MoE layer's 128 up and 128 down tensors from the shards. Quantize
  with the same call as `replace_parameter_4bit` (`F.quantize_4bit`, NF4, compress
  statistics), then dequantize. Record:
  - packed and statistics bytes;
  - the relative Frobenius round-trip error, per layer and overall;
  - the same error for that layer's `shared_experts` weights under `Linear4bit`
    quantization (the error standard QLoRA already accepts);
  - dequantize time for one full layer stack, as the median of 20 runs timed with CUDA
    events, expressed as GB/s of bf16 output.
- **Pass (CHOSEN):**
  - packed bytes are exactly numel/2;
  - statistics bytes are within 1% of the arithmetic above;
  - the overall expert round-trip error is ≤ 1.5× the shared-expert error.
- Speed is recorded, not gated. It explains G5's step time.
- **Result (MEASURED, 2026-09-24): PASS on both cards** (`results/g1-{laptop,desktop}.json`,
  probe `probes/g1_nf4_real_experts.py`). The per-layer numbers are identical on the two
  cards.
  - Packed: 14,687,404,032 B, exactly numel/2.
  - Statistics: 466,203,192 B against 466,152,960 arithmetic, +0.011%. The excess is
    1,092 B per stack × 46: the NF4 code (64 B), the offset (4 B) and the nested code
    (1,024 B), which the arithmetic above left out.
  - Round-trip error: experts 0.09238, shared experts 0.09587, ratio 0.964 (limit 1.5).
    Per expert 0.0890 to 0.0959.
  - Control: quantizing each expert on its own gives 0.09238 too (largest per-stack
    difference 1e-5). Fusing 128 experts into one tensor costs no precision.
  - The hook's call is the one tested: `replace_parameter_4bit(..., compress_statistics=True,
    quant_type="nf4")`, with blocksize `None` → 64 inside `F.quantize_4bit` (SOURCED raw from
    the image). The yardstick matches `~/gpu-lab/training/qlora.py` (NF4, double quant).
  - Speed: one layer's up + down dequantize (2.55 GB of bf16 out) takes a median 5.089 ms on
    the laptop (502 GB/s) and 4.284 ms on the 3090 (596 GB/s). ARITHMETIC: 23 layers ×
    5.1 ms ≈ 0.12 s of dequantize per forward on the laptop. G5 measures the real step
    cost, which includes the backward's second dequantize under checkpointing.
  - Reproduced (judge re-run on the laptop, `results/g1-judge-laptop.json`): per-layer
    numbers, bytes and checks bit-identical; dequantize 4.998 ms.

### G4: full load, idle VRAM (`probes/g4_route1_load.py`)
- **Smoke-tested 2026-09-24 on a tiny saved nemotron_h checkpoint in Lightning's own
  per-expert, `backbone.` layout (MEASURED):**
  - 4/4 expert stacks quantized by the hook;
  - `conv1d` correctly skipped;
  - 0 unquantized expert params;
  - 0 missing and 0 unexpected keys;
  - finite logits, and the paragraph loss at the random-init level
    (11.98 nats vs ln 151936 = 11.93).

  This proves the mechanism, not the real-size numbers.
- **Measures:**
  - load time (s);
  - host peak RSS (GiB);
  - torch peak allocated and reserved during load;
  - device-wide NVML peak, with the pre-CUDA baseline so that other apps can be
    subtracted;
  - idle allocated, reserved and NVML;
  - resident bytes by storage class, compared with the arithmetic;
  - missing and unexpected keys;
  - a sanity forward: mean NLL on a fixed paragraph, top-1 on three cloze prompts, and
    that forward's peak memory.
- **Pass (CHOSEN):**
  - the load finishes on the laptop without OOM, in RAM or VRAM;
  - 0 unquantized expert params;
  - 46 expert Parameters quantized by the hook;
  - no missing keys;
  - idle torch-allocated ≤ 18.0 GiB, which leaves ≥ 4.5 GiB for G5;
  - the sanity forward is finite, with " Paris" top-1 for "The capital of France is".

  A resident total more than 5% off the 16.15 GiB arithmetic is not a fail, but it is a
  surprise to explain before G5.
- Desktop: run the same probe. Its 31.9 GB of RAM is the thing to watch.
- **Result (MEASURED, 2026-09-24): PASS on both cards** (`results/g4-route1-{laptop,desktop}.json`).
  - 46/46 expert Parameters quantized by the hook; 0 unquantized 3D expert params; 23
    `conv1d` skipped; 0 missing and 0 unexpected keys; `grouped_mm`.
  - Idle torch-allocated 16.158 GiB (limit 18.0), 0.05% off the 16.15 GiB arithmetic, so
    there is no surprise to explain. By class: experts 13.679 packed + 0.434 statistics;
    116 `Linear4bit` modules 0.694 + 0.022; bf16 rest 1.329 (713,507,136 params).
  - Sanity forward finite; top-1 " Paris", " Au", " five".

  | | laptop (sm_120) | desktop (sm_86) |
  |---|---|---|
  | load time | 224.8 s (NFS) | 54.1 s (local disk) |
  | host peak RSS | 41.7 GiB | 25.2 GiB |
  | torch peak during load | 17.564 GiB | 17.564 GiB |
  | NVML: other apps before load | 1.556 GiB | 0.443 GiB |
  | NVML peak during load | 18.251 GiB | 18.231 GiB |
  | NVML idle | 18.056 GiB | 17.020 GiB |
  | forward (96 tokens, no grad): torch peak | 18.653 GiB | 18.629 GiB |
  | forward: NVML peak | 20.573 GiB | 19.524 GiB |
  | paragraph mean NLL | 2.0816 nats | 2.0714 nats |
  | max GPU temperature | 65 C | 52 C |

  - The NLL differs between the cards by 0.010 nats; the cause (per-architecture kernels)
    is not measured.
  - The desktop's 25.2 GiB RSS exceeded its container's 24 GiB limit without an OOM kill.
    RSS counts the memory-mapped shard pages, which are file-backed and reclaimable
    (inference, not verified).
  - ARITHMETIC, for G5: the laptop card is 23.89 GiB, so G5's line (card − 0.5 GiB) is
    23.39 GiB. The no-grad forward already peaks at 20.573 GiB, leaving 2.82 GiB for LoRA
    weights, optimizer state, activations and the backward. That includes 1.556 GiB of
    other apps' memory, which closing them would free. Whether it fits is UNKNOWN until G5.
  - Reproduced (judge re-run on the laptop, `results/g4-route1-judge-laptop.json`): hook
    counts, inventory, keys, idle and peak torch memory, NLL and top-1 identical. Host peak
    RSS was **50.2 GiB, not 41.7**, on a 61 GiB machine, so the RAM margin moves from run to
    run. The re-run started straight after G1 had read every shard, with 55 GiB of page
    cache warm; more mapped shard pages counted in RSS is the likely cause (not verified).
    Its other-apps baseline read 2.404 GiB, probably because G1's process had just exited.

### G2: forward fidelity vs bf16
- **Reference:** bf16 layer outputs computed one layer at a time on the GPU (one layer's
  bf16 weights at a time, ≤ 2.6 GB) on ~2k tokens of held-out text.
- **Two numbers per MoE layer:**
  - *isolated*: the NF4 layer on the bf16 input;
  - *accumulated*: the Route 1 model's hidden state against the reference.

  End to end: mean KL(bf16 ‖ 4-bit) in nats per token, top-1 agreement, and the NLL gap.
- **Yardstick:** the same end-to-end numbers for Qwen3-8B, bf16 against its NF4 load, on
  the same text. That is the degradation the lab already trains against.
- **Pass (CHOSEN):**
  - Lightning's mean KL is ≤ 2× Qwen3-8B's;
  - its top-1 agreement is ≥ Qwen3-8B's minus 5 points.
- If the streamed reference proves awkward (attention masks, mamba state), the fallback
  is a stock `device_map="auto"` bf16 load with CPU + disk offload for a single forward.
- **Probe details, fixed 2026-09-24 before the laptop run** (`probes/g2_fidelity.py`;
  thresholds above unchanged):
  - Text: the first 4 × 512 Lightning tokens of `~/gpu-lab/docs/phase2-dev-plane.md`
    (sha256 `971fa5f7…`, identical on both nodes), as four 512-token windows. Qwen sees
    the same characters, re-tokenized (494–507 tokens per window).
  - Why not one 2k sequence (ARITHMETIC, from the image's `mamba2_chunk_scan`): with no
    mamba kernels, the torch path materializes `C[..., None] * B[..., None, ...]` in fp32
    at (chunks, 128, 128, 64 heads, 128 state), which is 0.5 GiB per 128-token chunk:
    8 GiB at 2048 tokens, on top of the 16.16 GiB the model holds. 512 tokens is 2 GiB.
  - Reference: the model class on the meta device, one layer at a time materialized on
    the GPU and filled from the shards. The probe asserts that every tensor is filled, and
    compares dtypes against Route 1's unquantized tensors.
- **Debug run, desktop 3090, 2026-09-24 (MEASURED; not the gate, which is the laptop):**
  `results/g2-{lightning,qwen}-desktop-debug.json`.
  - Self-checks clean: 0 dtype mismatches over 239 unquantized tensors; embedding output
    identical (max diff 0.0); sdpa + grouped_mm on both paths; 46 stacks hooked.
  - Lightning 4-bit vs bf16: KL 0.1205 nats/token (p99 0.85, max 5.18), top-1 83.0%,
    NLL 2.937 → 2.998 (gap 0.061).
  - Qwen3-8B NF4 vs bf16: KL 0.0566, top-1 87.1%, NLL gap 0.018.
  - Against the CHOSEN lines: KL ratio **2.13 > 2.0, fail**; top-1 83.0 ≥ 82.1, pass.
  - Isolated per-layer update error by type, mean (max): attention 0.227 (0.456), mamba
    0.127 (0.166), MoE 0.107 (0.123). The worst layers are the NF4 attention
    projections, not the 4-bit experts.

### G5: training step (Route 1, placement A)
- **Setup:**
  - LoRA r=16, alpha=32, dropout 0;
  - gradient checkpointing, non-reentrant;
  - `enable_input_require_grads`;
  - no fp32 upcast of non-4-bit params, as with the lab's 32B precedent;
  - `paged_adamw_8bit`;
  - batch 1.
- **Sequence lengths:** 512, 1024 and 2048, using real samples from
  `~/gpu-lab/training/research_dataset_v3.json` (950 samples), rendered with Lightning's
  chat template.
- **Per length:** 3 warm-up steps, then 20 measured. Record:
  - torch peak allocated **per step**. A peak that grows step over step is the
    parametrize-cache leak that axolotl #3915 fixed in its own hooks; bnb 0.50.2's hooks
    carry the `always_call=True` fix (SOURCED raw), so a growing peak would be a surprise;
  - NVML peak;
  - step time (s), tokens/s, finite loss;
  - GPU temperature and throttle flags.
- **Thermal guard:** `max-power` profile. Abort if any `clocks_throttle_reasons` thermal
  or power-brake flag is set, or if the GPU reaches 87 C, the card's own target
  (memory: laptop-gpu-power-envelope).
- **Pass (CHOSEN):**
  - seq 1024 runs with NVML peak ≤ card total − 0.5 GiB;
  - one epoch of research_dataset_v3 at the measured seq-1024 tokens/s takes ≤ 8 h (one
    overnight run).
- Also record the mamba torch path's share of step time with the PyTorch profiler.
  Whether hub kernels exist for sm_120 is UNKNOWN; check kernels-community before
  claiming either way.
- **Probe details, fixed 2026-09-24 before the laptop run** (`probes/g5_train_step.py`;
  thresholds above unchanged):
  - Data: all 950 records render with Lightning's template and the lab's `TOOLS`
    (tool-call arguments parsed from JSON strings first); packed end to end and cut
    into exact L-token sequences, labels = inputs. One epoch = **933,904 tokens**
    (MEASURED; 766,480 if truncated at 1024 as `qlora.py` would).
  - The 8 h line therefore needs ≥ 933,904 / 28,800 s = **32.4 tokens/s** at seq 1024
    (ARITHMETIC). The epoch uses the untruncated count, the stricter of the two.
  - Device peak per step = the larger of the NVML reading, sampled every 0.1 s, and
    (NVML − torch reserved, just before the step) + the step's torch peak reserved. The
    second number cannot miss a short spike.
  - OOM at one L ends the sweep. The time breakdown = each block's checkpointed forward,
    recompute and backward, timed alone with CUDA events on real hidden states, plus a
    `torch.profiler` table of top CUDA ops for one full step.
  - **Risk (ARITHMETIC, not measured):** the same torch-path mamba product is 4 GiB at seq
    1024 in the forward, and its backward may hold two such products at once. With
    ~5.3 GiB of headroom (other apps open), seq 1024 may OOM on the torch path. That
    would be a Route 1 result as run. Kernels (a new dependency) or a memory-lean
    rewrite of the scan are David's call, not a retry.
- **Debug run, desktop 3090, 2026-09-24 (MEASURED; not the gate, which is the laptop):**
  `results/g5-desktop-debug.{json,log}`, 153.5 s wall, probe run unchanged.
  - Setup checks: 950/950 records rendered, epoch 933,904 tokens; 93 LoRA modules
    (in_proj and shared_experts up/down 23 each, q/k/v/o 6 each), 11,359,232 trainable
    params, none outside LoRA; 17.06 GiB NVML idle after PEFT.
  - seq 512: 20/20 measured steps, losses finite (3.16 at warm-up step 0, then 1.0–2.8),
    median 2.51 s = 204 tokens/s, torch peak allocated 20.78 GiB on every measured step,
    device peak 21.88 GiB (pass line 23.5). Warm-up step 0 sampled 23.81 GiB.
  - seq 1024: **OOM on step 0**, "Tried to allocate 4.00 GiB" with 21.32 GiB already
    allocated (23.56 GiB card). 4.00 GiB is exactly the fp32 product at
    `modeling_nemotron_h.py:298`, `(C[:, :, :, None] * B[:, :, None]).sum(dim=-1)`, at
    8 chunks × 128 × 128 × 64 heads × 128 state (ARITHMETIC; the probe keeps no traceback,
    so the op is inferred from the size). seq 2048 not run: an OOM ends the sweep.
  - Time at seq 512, each block's forward + recompute + backward timed alone: mamba
    1.748 s (70% of the step), MoE 0.725 s (29%), attention 0.035 s (1%). These sum to
    99.9% of the step, which leaves ~0 for lm_head, loss and optimizer, so the isolated
    times run slightly high and the shares are approximate. The profiler's top CUDA ops
    are elementwise mul, MulBackward0 and sum (the torch-path scan), ahead of mm,
    grouped_mm and bnb dequantize.
  - Thermal guard: one `sw_thermal` sample (phase max 65 C core) set `abort` during or
    just after the profiler step (still labelled `breakdown_seq512`); the per-block
    timing and the profiler table had both finished.
    `sw_power_cap` throughout, at the 350 W limit.
  - For the laptop gate (ARITHMETIC, not measured): torch sees 23.40 GiB there, less than
    the 3090's 23.56, and seq 1024 needed at least 25.32 GiB where it failed. The same
    code should OOM at seq 1024 on the laptop too, unless the torch path changes.
  - kernels-community (SOURCED raw, HF model API, 2026-09-24): `mamba-ssm` rev c8ffc584
    and `causal-conv1d` rev f2651e77 publish builds up to torch 2.12
    (`torch212-cxx11-cu130-x86_64-linux`); none for torch 2.13, which the image runs.
- **Result, Route 1 as run (MEASURED, laptop, 2026-09-24): FAIL**, on memory at seq 1024.
  `results/g5-laptop.{json,log}`, platform profile max-power, enforced power limit
  150 W, 1.52 GiB NVML in use by other apps before CUDA init.
  - seq 512: 20/20 measured steps, losses finite, median 2.595 s = 197.3 tokens/s, torch
    peak allocated 20.82 GiB on every measured step, device peak 23.01 GiB against the
    23.39 GiB line (0.38 GiB spare). Warm-up step 0 loss 3.19 (desktop 3.16).
  - seq 1024: OOM on step 0, "Tried to allocate 4.00 GiB" with 21.37 GiB allocated, as
    on the desktop. The gate reads "seq 1024 did not complete 20 measured steps".
  - Time at seq 512, blocks timed alone: mamba 1.912 s (74%), MoE 0.691 s (27%),
    attention 0.023 s (1%); the shares sum past 100%, so they run slightly high. No
    abort: max 71 C, 136 W, `sw_power_cap` only.
  - Per the decision rule: not feasible on one card as run. The memory-lean scan below is
    David's option 2; the pool stays the named fallback.
- **Lean scan check (MEASURED, desktop 3090, 2026-09-24):** `probes/lean_scan.py` rewrites
  the five broadcast-then-sum contractions in the torch `mamba2_chunk_scan` as
  `torch.einsum`; `probes/lean_scan_check.py` compares it to the original on random
  inputs at Lightning's shapes (seq 300 with initial states, seq 1024).
  - Line CHOSEN before the first run: every output and gradient within 1e-4 of the
    original's max. With bf16 inputs as the mixer passes them: **FAIL**, worst 1.39e-3
    (`results/lean-scan-check-desktop.json`). Every fp32 quantity agreed to ≤ 5.7e-7;
    only the gradients of bf16 inputs (hidden_states, dt, B, C) failed.
  - Same line with every input fp32, added after the fail (`...-fp32.json`): **PASS**,
    worst 7.75e-7 over outputs and all gradients. The math matches.
  - bf16 gradients described (`...-ulp.json`, not a pass line): at most 567 of 4,194,304
    elements differ, no sign flips; max abs difference ≤ 1.39e-3 of the tensor's max.
    A first reading, "each difference is one bf16 step", is refuted: the worst element is
    288 bf16 steps off. Why a few elements differ by many steps is UNKNOWN (suspected:
    near-zero elements where fp32 noise exceeds the bf16 step; untested).
  - The scan's own forward + backward peak: seq 1024 8.35 → 0.57 GiB, seq 2048 16.69 →
    1.14 GiB; time at 1024 0.109 → 0.027 s.
- **Route 1 + lean scan, desktop debug (MEASURED, 3090, 2026-09-24; not the gate):**
  `results/g5-lean-desktop-debug.{json,log}`, `probes/g5_lean_scan.py` (runs
  `g5_train_step.py` unchanged after `lean_scan.install()`); 3,381 scan calls went
  through the patch.
  - seq 512: 523 tokens/s (unpatched 204), device peak 20.26 GiB (unpatched 21.88).
  - seq 1024: 20/20 steps, 771.5 tokens/s, device peak 20.77 GiB, torch peak allocated
    19.64 GiB on every measured step. Probe's gate block: memory pass, epoch 0.34 h,
    pass.
  - seq 2048: 20/20 steps, 966.9 tokens/s, device peak 22.41 GiB.
  - Time at seq 1024: MoE 65%, mamba 31%, attention 3%. Profiler top ops: mm,
    grouped_mm, bnb dequantize.
- **Step-0 loss moved by the lean scan (MEASURED, desktop, 2026-09-24).** With LoRA B = 0,
  step 0 is the frozen model: unpatched 3.1572, lean 3.1861 on the same card and tokens.
  - Repeatable, not noise (`lean-scan-loss-check-desktop.json`, one load, five forwards):
    original 3.157189 twice with identical logits, lean 3.186051 twice. Lean against
    original: max logit difference 8.03, argmax changed at 42 of 512 positions.
  - Per call on real inputs (`lean-scan-layer-check-desktop.json`): lean within
    7e-8 to 4.3e-6 of the original's max over all 23 calls, so no bug on real data.
    Gaussian noise of the same std moved the loss only 0.003 and 0.007 (seeds 1, 2).
  - Against an fp64 scan (`lean-scan-fp64-check-desktop.json`): step-0 loss 3.1876.
    Per call, both are within 6.4e-6 of fp64, the original slightly closer on all 23
    calls (summed mean error 2.9e-7 vs 4.1e-7). At the loss, lean is 0.0015 from fp64
    and the original 0.030; the laptop's unpatched step 0 was 3.1907.
  - Reading: fp32-level rounding differences move this model's loss by up to ~0.03, and
    the unpatched desktop value is the outlier. Why the model is this sensitive is
    UNKNOWN (suspected: near-tie top-6 routing flips; untested).
- **Result, Route 1 + lean scan (MEASURED, laptop, 2026-09-24): PASS**, same CHOSEN lines.
  `results/g5-lean-laptop.{json,log}`, max-power, 150 W enforced, 1.52 GiB NVML used by
  other apps before CUDA init; 3,381 scan calls through the patch; no abort (max 73 C,
  147 W, `sw_power_cap` only).
  - seq 1024: 20/20 steps, 871.4 tokens/s, device peak 22.00 GiB against 23.39 (1.39 GiB
    spare), torch peak allocated 19.68 GiB on every measured step. Epoch 0.30 h
    against 8 h.
  - seq 512: 606.0 tokens/s, device peak 21.36 GiB. seq 2048: 1,070.5 tokens/s, but device
    peak 23.53 GiB is above the 23.39 line (not the gate length).
  - Step-0 loss 3.1787 (fp64-scan desktop 3.1876; see above).
  - Time at seq 1024: MoE 67%, mamba 28%, attention 3%.
  - Feasibility still needs G2 on the laptop (desktop debug failed the KL line).

### G7a: base Lightning on the lab's evals (cheap side check)
- Serve `...-NVFP4` (20.1 GiB) with vLLM 0.29 on the laptop. Run the same eval sets that
  the Qwen3-8B adapters lost to base + catalog.
- Whether NVFP4 runs on sm_120 under vLLM 0.29 is UNKNOWN; this gate settles it.
- **Needs another download (20.1 GiB), so it needs David's go.** Recommended before G5's
  long runs: if base Lightning already wins those evals, G6's question changes.
- **Download (authorised 2026-09-24, "download G7"): done and verified.** G7 serves the
  same checkpoint, so this covers both.
  - Pinned rev `bee75962`: 70 files, 21,583,785,438 B (20.10 GiB), 55 LFS. The manifest
    is `results/g7-nvfp4-manifest.json` (SOURCED raw, HF tree API).
  - Fetched like G0: on the desktop, onto its local disk, as uid 1000, with
    `hf download --revision` inside `gpu-lab:training`. It took 6 min 14 s.
  - **Download check (MEASURED): PASS.** This is not the G7a gate, which is still unrun
    (`results/g7-nvfp4-verify.json`, `probes/g0_verify.py` run on the desktop host). 70/70 files at manifest size; 55 sha256 and 15 git-blob hashes
    match; every file is uid 1000, mode 0644, so it is readable over NFS. All 70 sizes
    were also checked from the laptop over NFS.

### G6: short training + held-out eval
- Train placement A on research_dataset_v3 with the same epochs, rank and data as the
  Qwen3-8B v3 adapter, so the comparison is like for like.
- Evaluate with `~/gpu-lab/bench/research_eval.py`. It talks to a vLLM server
  (`/tokenize` + `/v1/completions`), **so G6 depends on G7's LoRA serving.**
- **Pass:**
  - the adapter beats base Lightning on the `held_out` split;
  - it does not regress `trap`, `trap_control` or `no_tool`;
  - it does not regress the task rows where every 8B adapter lost to base: operator/farm
    tasks, alert rules and live Prometheus + catalog (memory: research-eval-8b-results).
    Their harness is not located yet; find it in `~/gpu-lab/bench` before G6.

### G7: serving on vLLM 0.29 (laptop)
- Base NVFP4, then base + LoRA, then merge if needed. UNKNOWN:
  - whether vLLM's nemotron_h supports LoRA on placement A's modules (read the vLLM
    source before G6);
  - how much the train/serve precision mismatch (NF4 while training, NVFP4 while serving)
    costs. G6 measures that cost, because its eval runs through the server.
- Merge path (fallback): merge the LoRA into BF16, then re-quantize. Which tool does that
  for nemotron_h is UNKNOWN.

## Hard stops and rules

- No edits to `~/gpu-lab` until the feasibility verdict. Probes and results stay in this
  repo.
- No new dependency or image change without David's OK. That includes Route 2's
  `mamba_ssm` and axolotl.
- Every download beyond G0 needs David's go.
- Thermal guard as in G5. Any OOM or throttle is recorded as a result, not retried with
  moved thresholds.

## Why it might not matter

In the lab's evals the Qwen3-8B adapters cut both ways (memory: research-eval-8b-results,
base with thinking off vs v3):
- They won the reflex they trained on: held-out flag questions went from 69% to 99%.
- Base beat every adapter on tasks:
  - operator/farm tasks: 85/60% vs 55/25%;
  - alert rules checked with promtool: 78% vs 0%;
  - live Prometheus + catalog: 94% vs 50%.

QLoRA also did not inject facts (memory: qlora-does-not-inject-facts). That is why G7a
exists, why G6 gates on the task rows as well as the flag rows, and why "worth it" is a
separate verdict from "feasible".

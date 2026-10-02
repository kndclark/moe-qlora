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
- **Verdict (David, 2026-09-25): feasible**, in the configuration Route 1 + lean scan +
  attention q/k/v/o in bf16: G2 1.903x as run and 1.908x lean (line 2x), G5 22.10 GiB at
  seq 1024 (line 23.39), G1 and G4 unaffected (see each gate). Route 1 as defined stays
  recorded as failed on G2 and G5. The G2 margin is thin (5%, one text). "Worth it" is
  still open and needs G6. The `~/gpu-lab` edit ban under "Hard stops" is lifted.

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
  - From the laptop's local mirror (2026-09-27, `results/g4-route1-mirror-laptop.json`,
    via `probes/gpurun.sh`, page cache dropped first): **load 17.3 s, not 224.8 s over
    NFS**. Torch peak 17.564 GiB, NLL 2.0816 and top-1 identical; host peak RSS 38.5 GiB.

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
- **Result (MEASURED, laptop, 2026-09-25): FAIL** (`results/g2-{lightning,qwen}-laptop.json`).
  Max-power profile; self-checks as on the desktop (0 dtype mismatches over 239 tensors,
  embedding diff 0.0, 46 stacks hooked).
  - Lightning: KL 0.1208 nats/token (p99 0.80, max 5.09), top-1 83.27%, NLL gap 0.063.
  - Qwen3-8B: KL 0.0562, top-1 86.83%, NLL gap 0.017.
  - KL ratio **2.148 > 2.0, fail**; top-1 83.27 ≥ 81.83, pass. The gate needs both.
  - Isolated error by type, mean (max): attention 0.227 (0.457), mamba 0.127 (0.167),
    MoE 0.107 (0.123); the same ranking as the desktop.
  - Input: the doc now carries an uncommitted Ally X to-do, so `doc_sha256` reads
    `7a880f38…` / 4448 tokens against the debug run's `971fa5f7…` / 4089. The edit is
    past the first 2048 tokens; `predicted_tokens` (2044 / 2004) match the debug runs.
- **Route 1 + lean scan (David asked 2026-09-25), laptop: FAIL**
  (`results/g2-lean-lightning-laptop.json`, `probes/g2_lean_scan.py`; the patch covers
  both the bf16 reference and Route 1, 276 scan calls = 23 x 4 x 3 as expected). KL
  0.1198, ratio **2.130 > 2.0, fail**; top-1 83.66%, pass. Against the same Qwen run,
  which has no mamba layers. The scan choice moves the ratio by 0.019; neither passes.
- **Route 1 + attention in bf16 (follow-up David authorised 2026-09-25), laptop: PASS
  on both scans.** A new configuration beside the gate result above, which stays failed.
  `probes/attn_bf16.py` wraps the probes unchanged and adds `llm_int8_skip_modules` for
  the 24 attention q/k/v/o_proj and lm_head (an explicit list replaces the default
  lm_head skip). It changes nothing else: 92 Linear4bit layers and the 46 expert stacks
  stay NF4. Cost +0.194 GiB (ARITHMETIC, 6 x 23,396,352 params, bf16 vs NF4). NVIDIA's
  own NVFP4 release of this model keeps the same 24 projections in bf16
  (`hf_quant_config.json` ignore list, rev `bee75962`, read from the local snapshot).
  - As run (`results/g2-attnbf16-lightning-laptop.json`): KL 0.1070 nats/token (p99
    0.71), ratio **1.903 ≤ 2.0, pass**; top-1 84.20%, pass; NLL gap 0.049.
  - With the lean scan (`results/g2-attnbf16-lean-lightning-laptop.json`, 276 scan
    calls): KL 0.1073, ratio **1.908, pass**; top-1 84.10%, pass.
  - Self-checks: 0 dtype mismatches over 263 unquantized tensors (239 + the 24 attention
    weights), embedding diff 0.0, isolated attention error exactly 0.0 (bf16 weights
    are the reference's), mamba 0.127 and MoE 0.107 unchanged. Same Qwen yardstick
    (`g2-qwen-laptop.json`, every linear layer NF4).
  - The margin is thin: 5% under the line, on one text of 2,044 predicted tokens.

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
- **Result, Route 1 + lean scan + attention in bf16 (MEASURED, laptop, 2026-09-25):
  PASS**, same CHOSEN lines; the configuration that passed G2 above.
  `results/g5-attnbf16-lean-laptop.{json,log}` via `probes/attn_bf16.py g5_lean_scan.py`
  (0 attention Linear4bit, 24 bf16, 92 Linear4bit in all; 3,381 scan calls). Max-power,
  175 W enforced this time (150 W on 09-24), but the card drew at most 148.5 W in either run,
  so the limit did not bind; 1.43 GiB NVML used by other apps before CUDA init; no abort.
  - seq 1024: 20/20 steps, 856.6 tokens/s (-1.7%), device peak 22.10 GiB against 23.39
    (1.30 GiB spare), torch peak allocated 19.88 GiB: +0.194 GiB, as the arithmetic said.
    Epoch 0.30 h against 8 h.
  - seq 512: 593.4 tokens/s, device peak 21.48 GiB. seq 2048: 1,066.8 tokens/s, device
    peak 23.82 GiB, now 0.43 GiB over the line (was 0.14; not the gate length).
  - Step-0 loss (seq 512, LoRA B = 0, the frozen model) 3.1403 against 3.1787 with 4-bit
    attention. Lower, but no bf16 reference loss exists for these tokens, so "closer to
    bf16" is not measured here; G2's KL is the measure.
  - Time at seq 1024: MoE 66%, mamba 28%, attention 2%.
  - G4's idle line (torch-allocated ≤ 18.0 GiB) holds for this configuration on this
    run's own load: 16.39 GiB after PEFT, against 16.20 for the lean-scan run above.
    `g4_route1_load.py` itself was not re-run.

### After the verdict: memory and placement follow-ups (David, 2026-09-25)
David asked for all three code-only ideas from the NVIDIA research. Each runs on the
verdict configuration (Route 1 + lean scan + attention bf16) and is recorded beside it.
None of them changes a gate result above.

**F1: chunked cross-entropy, to fit seq 2048 on the laptop.** `probes/chunked_ce.py` runs
lm_head and the loss a chunk of positions at a time (default 256), each chunk under
checkpointing, so neither the full bf16 logits nor their fp32 copies exist. The idea is
NVIDIA AutoModel's `_ChunkedCrossEntropySum`, which chunks only the fp32 upcast.
- Step 1, where the peak is (`probes/g5_mem_phases.py`, seq 2048): torch peak per phase
  with the stock loss, then with the chunked loss, plus a CUDA memory history at the peak.
- Exactness, loss head alone on real final hidden states (CHOSEN before the run):
  |loss difference| ≤ 1e-4 nats; gradient w.r.t. the hidden states max |difference| ≤ 1e-2
  × max |stock gradient| and cosine ≥ 0.9999. (CPU fp64 toy check before the run: loss
  4.8e-7 apart, both paths computing in fp32; gradients 1.7e-18 apart.)
- Pass (CHOSEN before the run): G5's own lines, at seq 2048: 20/20 measured steps, finite
  losses, device peak ≤ card − 0.5 GiB. Seq 1024 must still pass G5. Speed is recorded,
  not gated.
- **Step 1 result (MEASURED, laptop, 2026-09-25; `results/g5mem-attnbf16-lean-2048-laptop.
  {json,log}`):** the peak was at the loss, and chunking removes it.
  - Stock loss: step torch peak **21.03 GiB, in the loss backward** (body forward 19.76,
    body backward 20.90). Live then, of the step's own allocations (4.55 GiB): fp32 logits
    from `modeling_nemotron_h.py:1170` 1.0 GiB, `fixed_cross_entropy` 1.0 GiB, two
    frame-less 1.0 GiB buffers (the backward's fp32 gradients, by size), checkpointed layer
    inputs 0.52 GiB.
  - Chunked (256): step peak **19.90 GiB, 1.12 GiB lower**, now in the body backward: the
    two dequantized expert stacks (`dequantize_4bit`, 2.38 GiB) plus the checkpointed
    inputs. Loss forward 17.29 and loss backward 17.42 (stock 19.04 and 21.03). The stock
    body backward is exactly 1.0 GiB higher because the returned output keeps its fp32
    logits alive through the backward.
  - Exactness: **pass.** Loss 1.612694 on both (difference 0.0); gradient max difference
    7.2e-7 against a max of 9.5e-5 (0.75%, line 1%; about 1.5 bf16 steps at that size),
    cosine 0.999996. Not bitwise equal: the lm_head backward runs as 8 smaller matmuls.
  - No abort; max 69 C, 141 W, `sw_power_cap` only; 1.53 GiB NVML used by other apps.
- **Result (MEASURED, laptop, 2026-09-25): PASS at seq 2048.**
  `results/g5-attnbf16-lean-cce-laptop.{json,log}` via `probes/attn_bf16.py
  g5_chunked_ce.py` (`g5_train_step.py` unchanged; 3,381 scan calls, 70 chunked-loss
  calls, 24 bf16 attention projections). Max-power, 150 W enforced (175 W in the run
  compared against), 1.58 GiB NVML used by other apps before CUDA init; no abort, max
  74 C, 146 W, `sw_power_cap` only.

  | seq | device peak, GiB (line 23.39) | torch peak, GiB | tokens/s |
  |---|---|---|---|
  | 512 | 21.13 (was 21.48) | 19.11 (19.36) | 581.0 (593.4, −2.1%) |
  | 1024 | 21.41 (22.10) | 19.38 (19.88) | 841.1 (856.6, −1.8%) |
  | 2048 | **22.09 (23.82, over)** | 19.90 (21.03) | 1,029.9 (1,066.8, −3.5%) |

  - 20/20 measured steps at every length, losses finite, torch peak flat step to step.
    Seq 1024 still passes G5: epoch 0.31 h.
  - Step-0 loss at seq 512 3.1403, identical to the unchunked run. Later lengths start
    after 23+ optimizer steps, so theirs differ slightly (1.3686 vs 1.3682, 0.9649 vs
    0.9693); gradients are not bitwise equal (see exactness), so that is expected.
  - The slowdown is the lm_head recompute plus, possibly, the lower enforced power limit
    (146 W drawn against 150); this run cannot separate the two.
  - Time at seq 1024: MoE 68%, mamba 28%, attention 2%.

**F2: a leaner LoRA backward.** `probes/lean_lora.py` replaces the plain path of PEFT 0.21's
`Linear4bit` and `Linear` LoRA branch with one autograd Function that saves the bf16 input
and the two factors and recomputes `x.float() @ A^T` in the backward. PEFT's own branch
(adapters fp32 by its default `autocast_adapter_dtype=True`) saves an fp32 copy of the
input and the fp32 A-activation. The idea is NVIDIA AutoModel's `LoRATritonFunction`
(Triton there, plain PyTorch here). Same ops in the same order as PEFT's forward.
- Pre-run check (MEASURED, `results/f2-check.{json,log}`, one 2688 → 3712 layer, r=16):
  forward and all three gradients (input, A, B) **bitwise equal** to PEFT's, on CPU (64
  tokens) and on the laptop GPU (2,048 tokens), for both `Linear` and `Linear4bit`. What
  the graph keeps per layer at 2,048 tokens: PEFT 21.5 MiB, lean 10.5 MiB (the bf16 input,
  which in the model another op often keeps anyway).
- Expectation (ARITHMETIC, not a line): under checkpointing only the layer being
  recomputed holds these, so the step peak falls by 0 to about 0.035 GiB at seq 2048. It
  may be 0: the peak is in the routed experts' backward, and autograd probably runs the
  shared experts' backward (created later in the forward) before it.
- Pass (CHOSEN before the run): G5's own lines at seq 1024 and 2048, as F1; `lean_lora`
  calls > 0 for both kinds and 0 fallbacks; step-0 loss at seq 512 exactly F1's 3.1403
  (the forward is bitwise PEFT's). Reported, not gated: torch and device peak against F1
  at each length, tokens/s.
- Run: `EXPERT_LORA` unset, `LEAN_LORA=1`, `probes/attn_bf16.py g5_followups.py LABEL`
  (`g5_followups.py` = F1's configuration plus switches; `g5_train_step.py` unchanged).
- **Result (MEASURED, laptop, 2026-09-25): PASS, and no gain.**
  `results/g5-attnbf16-lean-cce-leanlora-laptop.{json,log}`: 10,143 `Linear4bit` and 3,528
  `Linear` calls through the Function, 0 fallbacks; 150 W enforced, 1.44 GiB NVML used by
  other apps; no abort, max 74 C, 145 W, `sw_power_cap` only.

  | seq | device peak, GiB (F1) | torch peak, GiB (F1) | tokens/s (F1) |
  |---|---|---|---|
  | 512 | 21.15 (21.13) | 19.114 (19.114) | 585.0 (581.0) |
  | 1024 | 21.48 (21.41) | 19.377 (19.377) | 841.1 (841.1) |
  | 2048 | 22.09 (22.09) | 19.901 (19.901) | 1,025.5 (1,029.9) |

  - The torch peak is **identical to the MiB** at every length, so the saving at the peak
    is zero. Device peaks differ by other apps' NVML use (idle after PEFT: torch numbers
    identical, NVML 18.23 vs 18.17 GiB). Step-0 loss 3.1403, as required.
  - Later losses differ slightly from F1's (step 1 at seq 512: 1.984 vs 1.988) although
    the isolated check was bitwise. Either the step is not reproducible run to run, or the
    gradient contributions into a shared input are summed in another order. UNKNOWN which;
    an unchanged rerun of F1 would settle it, and it also bears on F1's own reading of its
    later-step loss differences.
  - Not worth keeping: NVIDIA's version pays off by fusing kernels and when training
    without checkpointing, and neither applies here.

**F3: LoRA on lm_head and the routed experts.** Correction to how this was proposed: no
NVIDIA recipe targets lm_head on purpose (SOURCED: AutoModel's exclude-mode default
`["*.out_proj"]` would catch it only incidentally; its VL recipes exclude it), and
Megatron-Bridge's Lightning recipe targets the expert linears with
`share_expert_adapters=True` by default: one adapter shared by every expert in the layer,
not one per expert. AutoModel's Nemotron recipes put no LoRA on experts.
- Code: `probes/expert_lora.py` (drafted by a research agent, shared mode added) adds the
  LoRA to the activations inside the experts forward, so no (128, out, in) delta is built
  and the NF4 expert weights stay as they are. Stock PEFT 0.21 `target_parameters` crashes
  here (hand-off #4). lm_head (bf16, untied) takes an ordinary PEFT LoRA and, with F1's
  chunked loss, runs a chunk at a time.
- Pre-run checks (MEASURED, `results/f3-check.{json,log}`, CPU fp32, tiny nemotron_h):
  with B = 0 the LoRA experts forward is **bitwise** the stock `grouped_mm_experts_forward`
  in both modes; with random factors, output 6.2e-7 (per-expert) and 7.4e-7 (shared)
  relative to a dense fp32 reference, factor gradients ≤ 1.1e-6. Whole tiny model,
  checkpointing on, placement A + lm_head + experts: chunked vs stock loss 0.0 / 4.8e-7
  apart, every gradient present and nonzero, worst 6.4e-7 relative. GPU, real shapes,
  bf16: per-expert r=16 and r=8 run (0.4-0.5% from an fp32 loop, bf16 rounding); **r=4
  cannot run**: `torch` grouped_mm needs 16-byte strides ("strides should be multiple of
  16 bytes"), so r=8 is the smallest per-expert rank on this path.
- Configurations, all on F1's (placement A unchanged, r=16), all with lm_head r=16, alpha
  = 2r throughout:

  | config | expert factors | params, M | resident, GiB (ARITHMETIC) |
  |---|---|---|---|
  | shared r=16 | fp32, one set per layer | 3.34 + lm_head 2.14 | 0.05 |
  | per-expert r=16 | bf16 (fp32 would be 3.99 GiB) | 428.1 + 2.14 | 2.41 |
  | per-expert r=8 | bf16 | 214.0 + 2.14 | 1.22 |

  Resident = weights + gradients + 8-bit Adam (2 bytes). Headroom under the line on F1:
  1.98 GiB at 1024, 1.30 at 2048. So (ARITHMETIC) shared passes both lengths; per-expert
  r=16 fails at 1024 (about 23.9 GiB incl. the agent's +0.10 GiB transient, against
  23.39, and the card has 23.89); r=8 passes at 1024 (about 22.7) and misses at 2048 by
  about 0.1. UNKNOWN: the Adam states are bitsandbytes *paged* memory, which the driver can
  move to host RAM under pressure; that could shift both peak and speed.
- Order (CHOSEN): shared r=16; per-expert r=16; per-expert r=8 only if r=16 fails at 1024.
- Pass, per configuration (CHOSEN before the runs): G5's lines at seq 1024 (20/20 measured
  steps, finite losses, device peak ≤ card − 0.5 GiB, epoch ≤ 8 h); seq 2048 against the
  same memory line, reported separately; step-0 loss at seq 512 exactly F1's 3.1403 (every
  new B starts at 0 and adds exact zeros); by the end of the run every expert factor
  tensor (23 × 4) and both lm_head factors have moved from their initial values at the
  sampled positions, else the adapter did not train. F2 stays off, so each F3 result
  differs from F1 by one change.
- Run: `EXPERT_LORA=shared|per_expert EXPERT_R=16|8 LM_HEAD_LORA=1`, same command as F2.
- **Results (MEASURED, laptop, 2026-09-25): shared r=16 PASS, per-expert r=16 FAIL
  (memory), per-expert r=8 PASS**, each at both 1024 and 2048.
  `results/g5-attnbf16-lean-cce-f3{shared16,perexpert16,perexpert8}-laptop.{json,log}`.
  All three: 20/20 measured steps at every length, losses finite, step-0 loss 3.1403 at
  seq 512, all 92 expert factor tensors and both lm_head factors moved, 3,381 LoRA experts
  calls, no abort, 150 W enforced, max 75 C, `sw_power_cap` only, 1.38-1.43 GiB NVML used
  by other apps.

  | config | trainable, M | device peak 1024 / 2048, GiB (line 23.39) | torch peak 1024 / 2048 | tokens/s 512 / 1024 / 2048 | epoch h @1024 |
  |---|---|---|---|---|---|
  | F1 (reference) | 11.36 | 21.41 / 22.09 | 19.38 / 19.90 | 581.0 / 841.1 / 1,029.9 | 0.31 |
  | shared r=16 | 16.84 | 21.57 / 22.25 | 19.44 / 19.99 | 588.9 / 844.0 / 1,026.7 | 0.31 |
  | per-expert r=16 | 441.58 | **23.82 / 23.82** | 20.80 / 21.12 | 407.2 / 674.4 / 900.2 | 0.38 |
  | per-expert r=8 | 227.54 | 22.40 / 22.95 | 20.02 / 20.43 | 431.4 / 707.4 / 930.0 | 0.37 |

  - Shared r=16 costs almost nothing: +0.16 GiB device peak, speed unchanged.
  - Per-expert r=16 did **not** run out of memory: the device sat at 23.82 of 23.89 GiB at
    every length, i.e. full, while torch's own peak rose only 1.43 GiB at 1024 (arithmetic:
    2.41 resident). The 8-bit Adam states are bitsandbytes paged memory, outside torch's
    allocator; with the card full the driver probably kept part of them in host RAM, which
    would also account for part of the slowdown. Page traffic was not recorded, so the
    paging is inferred, not measured. It fails G5's memory line either way.
  - Per-expert r=8 fits 2048 with 0.44 GiB to spare where the arithmetic said it would
    miss by 0.13: the arithmetic counted every gradient and Adam state as resident at the
    peak (measured torch rise +0.64 GiB at 1024, arithmetic 1.22).
  - The per-expert cost in time is the extra grouped matmuls, not the rank: MoE blocks take
    1.04 s per step at 1024 at both r=8 and r=16 (F1 0.82 s); −16% tokens/s at 1024, −26%
    at 512. Epoch time is still far under the 8 h line.
  - Training loss falls further with more adapter capacity (mean of the last 5 steps at
    seq 2048: F1 0.667, shared 0.604, per-expert r=8 0.550, r=16 0.497). This is loss on
    the training stream, not an eval, and could be memorisation; only G6 can say whether
    any of it helps.
  - Open before G6/G7 (UNKNOWN): whether vLLM 0.29 can serve either expert layout (the
    fallback is folding `dense_delta` into the BF16 experts and re-quantizing); the
    per-expert factors are bf16 without fp32 master copies, so 8-bit Adam's small updates
    at lr 1e-4 may be partly rounded away.

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
- **G7a run — CHOSEN before the run, 2026-09-25 (David: "affirmative ... proceed").**
  - Serve (laptop): `vllm/vllm-openai:v0.29.0`, NVFP4 rev `bee75962`, flags from the model
    card's closest config (1x DGX Spark GB10, the card's only consumer-Blackwell entry,
    validated there on v0.27.1): `--kv-cache-dtype fp8 --mamba-backend flashinfer
    --mamba-cache-mode align --moe-backend marlin`, minus speculative decoding and MTP;
    plus `--max-model-len 16384 --max-num-seqs 16` and a `--gpu-memory-utilization` set
    from free memory at start. Any flag change needed to make it load is recorded here.
  - Harness: `probes/g7a_eval.py` = `~/gpu-lab/bench/research_eval.py` as committed
    (64919bd, repo HEAD 603c317) with two reading patches, scorer untouched: Lightning's
    XML tool-call form, and "no `</think>`" = truncated when the prompt pre-opens
    `<think>`. `--selfcheck` must PASS first. vLLM json-loads tool arguments before
    templating (SOURCED: vllm 0.29.0 `entrypoints/chat_utils.py:2043`), so no patch there.
  - Primary runs: the yardstick's exact settings (`--thinking off --max-tokens 512
    --max-calls 3 --temperature 0 --window 4000 --seed 20260923`, concurrency 16) on all
    seven sets: v1, v2, rocky, promql `--promql-catalog`, general, alert, trap3. Output
    `results/research-eval-<set>-lightning-nothink.json`.
  - Yardstick: Qwen3-8B base, thinking off (`bench/results/research-eval-{L,v2-L,rocky-L}-base-
    nothink.json`, `-{promqlcat,general,alert,trap3}-8b-base-nothink.json`). Reference:
    the v3 adapter, thinking off (`-L-adv3-nothink` of each set).
  - Secondary (descriptive, no rule): v1 with `--thinking on --max-tokens 4096` against
    `research-eval-8b-base-think-4k.json`.
  - **Serving gate:** PASS = the server loads and all seven sets finish with 0 items in
    status `error` and 0 turns whose text holds `<tool_call>` that the parser missed.
    Anything else FAILS and is recorded with its output. This settles NVFP4 on sm_120.
  - **Comparison rule:** headline metrics per split — `hit_and_grounded` on flag/task
    splits; `denied_heuristic` on trap splits (higher is better) and on trap_control
    splits (lower is better); `over_trigger` (lower) and `correct_where_scorable` on
    no_tool/general; `correct` on alert and promql; `noticed` (higher) and `fabricated`
    (lower) on trap3. A difference is a win or a loss only when it exceeds 4 items on that
    split (the base noise floor, MEASURED 2026-09-23); otherwise it is a tie. Small splits
    (alert 9, trap3 12, promql 18) therefore rarely show a win; that is accepted.
  - **Reading (fixed now):**
    (a) Lightning base ties or beats the v3 adapter on `held_out`, `held_out2` and
    `rocky_held_out` -> the one job the 8B adapters won needs no adapter on Lightning;
    G6's question narrows to the task rows (task, rocky_task, alert, promql) and traps.
    (b) Lightning base loses to Qwen3-8B base on at least half the headline splits ->
    tell David before G6: Lightning may be the weaker default base for this lab's evals.
    (c) Otherwise G6 proceeds as planned.
  - Also recorded: load time, GPU memory after load, and each set's elapsed seconds.
- **G7a serving — flag changes needed to load (MEASURED, 2026-09-25; logs
  `results/g7a-serve-attempt{1..6}*.log`):**
  1. `--enforce-eager` added: with CUDA graphs, vLLM budgeted ~2.2 GiB for them and left
     0.11 GiB of KV, under the 0.14 GiB one 16384-token request needs (attempt 1).
  2. `--gpu-memory-utilization` 0.91 -> 0.85 (attempts 2-3): did not fix the crash below,
     kept for headroom; KV is still 1.5 GiB = 170,666 tokens.
  3. `--mamba-backend flashinfer` dropped: FlashInfer's
     `selective_state_update_kernel_producer_consumer_vertical<bf16,...>` fails
     `cuLaunchKernel` with CUDA_ERROR_OUT_OF_MEMORY on sm_120 (CUDA_LOG_FILE=stderr,
     attempt 6; nvidia-smi trace peak 23.01 of 23.89 GiB, attempt 5); torch reports it
     one op later at `mamba_mixer2.py:116`. CORRECTED 2026-09-26: this line used to say
     "while 1.5 GiB of the card is free"; that 1.5 GiB is vLLM's "Available KV cache
     memory" (attempt 6 log line 126), a budget, not free card memory. The launch needs
     ~1.6 GiB free for a driver stack reservation; see "Side quest: FlashInfer SSU". The default
     Triton SSU backend works.
  - Result: NVFP4 **serves on sm_120 under vLLM 0.29** (MoE via MARLIN weight-only FP4:
    "Your GPU does not have native support for FP4 computation"). Weights 17.86 GiB; 79 s
    to ready with the files in page cache (76 s weight load alone over NFS, cold);
    23.34 GiB used on the card after load, 0.62 GiB free.
  - Harness smoke (2 per split, thinking off): 6/6 `<tool_call>` turns parsed as
    `xml_function`; 4/10 answers hit the 512-token cap (Lightning answers long).
  - Output name: the promql-with-catalog set is saved as `promqlcat` (bench's convention).
- **G7a RESULT (MEASURED, 2026-09-25; `results/research-eval-*-lightning-*.{json,log}`):**
  - **Serving gate: PASS.** 8 runs, 636 items, 0 in status `error`; 488 tool calls, all
    in Lightning's XML form, 0 `<tool_call>` turns missed. Elapsed 6-116 s per set
    thinking off, 207 s for v1 thinking on.
  - **Comparison (thinking off, CHOSEN settings) vs Qwen3-8B base: 0 wins, 8 losses,
    14 ties** over 22 headline metrics (19 splits). Losses, in items: held_out −18,
    held_out2 −17, rocky_held_out −13, promql −11, seen_tool −5, trap2 −5 (denied 0/12
    asserted fakes), rocky_trap −5, trap3 noticed −5. Ties include task +1, rocky_task −1,
    alert −2, general −1, no_tool 0. vs the v3 adapter it WINS task, rocky_task, alert
    (where v3 collapsed) and loses held_out, seen_tool, held_out2, two_flag,
    rocky_held_out and all three trap splits (fix_cmd ties).
  - **Reading, by the rule:** (a) not met (loses to v3 on all three held-out splits);
    (b) not triggered (8 of 19 splits lost, under half); so **(c): G6 proceeds as
    planned.**
  - **What drives the thinking-off losses (MEASURED diagnostics, not part of the rule):**
    - It rarely looks up: held_out lookups 8/90 vs Qwen 60/90.
    - The 512-token cap cuts its long answers: truncated v1 48/158, v2 46/134, rocky
      35/102 (Qwen base: 1/158, 1/134, 2/102). On held_out, the 58 untruncated items still lose
      (36 vs Qwen 43); the 32 truncated ones lose more (10 vs 21).
    - promql: 16 of 28 calls were `bash`, 14 of them refused by the harness's
      `--help`/`man` allowlist (`cat /sys/class/power_supply/...`, the local machine); Qwen used `promql`
      20/20. The bash tool's description invites "examine system status", so part of
      this loss is the harness's contract, not only the model.
  - **Secondary, thinking on, 4096 tokens (descriptive):** held_out 0.844 (lookups 67%)
    vs Qwen3-8B thinking 0.656 and Qwen base thinking-off 0.711; seen_tool 0.600 vs
    0.520; trap denied 0.533 vs 0.800; no_tool over-trigger 0.20 vs 0.35; 1/158
    truncated. With thinking, Lightning is the better flag-looker of the two bases, still
    under v3 (1.000), and weaker at denying fakes.
  - For G6 (UNKNOWN until David decides): the thinking mode moves held_out by 33 points
    on the base, so G6 must fix its thinking mode before its run; the v3 comparison
    rows were thinking off.

### Side quest: FlashInfer SSU on sm_120 (David asked, 2026-09-26)
Question: is G7a's `--mamba-backend flashinfer` crash a software gate we can get around,
or a hardware limit? **Answer: software** (a toolchain lowering, not a gate); a local
13-call-site header patch makes it serve. Harness `probes/ssu_sm120/`, logs
`results/ssu_sm120/`.
- **Cause (MEASURED unless tagged):** ptxas (CUDA 13.0.88 and 13.3.73) lowers the
  `.shared::cluster` form of the TMA load `cp.async.bulk.tensor.4d...global.tile.mbarrier::
  complete_tx::bytes` to a driver syscall (`__cuda_syscall_cp_async_bulk_tensor_4d_tile_
  unicast`) on sm_120/120a/120f; on sm_90a/sm_100a it is inline. The syscall lifts the
  per-thread stack 1,024 -> 14,608 B; the drop in free memory at the first launch of any
  kernel containing it is 1,634 MiB (a control kernel whose load never runs costs the
  same) = the 13,584 B increase x 82 SMs x 1,536 threads = 1,631.7 MiB (ARITHMETIC; the
  full 14,608 B would be 1,754.7). Unpatched launch works at >= 1,689.7 MiB free, fails at
  <= 1,679.1 MiB. That too little was free at that moment under vLLM (0.85 utilisation)
  is inferred from the OOM, not measured.
- The form comes from libcu++ `cuda::device::experimental::cp_async_bulk_tensor_4d_
  global_to_shared`, which hardcodes the cluster space (SOURCED: the JIT compiles
  FlashInfer's bundled CCCL 3.3.2, `flashinfer/data/cccl/libcudacxx/include/cuda/
  barrier:164-182`, per its build.ninja; the CUDA 13.0 toolkit copy at :151-169 is the
  same). libcu++ already ships the `.shared::cta` form of the load
  (`cp_async_bulk_tensor.h:639`). FlashInfer 0.6.18 calls the helper 13 times in
  `flashinfer/mamba/kernel_selective_state_update_{stp,mtp_vertical,mtp_horizontal}.cuh`.
  The nightly image's SSU code is identical; v0.7.0 was compared for the stp header only.
  Newer nvcc (13.3) does not fix it. Whether upstream has a fix or an issue: UNKNOWN
  (not searched online).
- Refuted: shared memory too big (25,728 of 101,376 B); the kernel's own stack (8 B);
  duplicate kernel copies in the .so (real, harmless).
- **Patch (local, image unchanged):** a guarded inline-asm shim issuing the `.shared::cta`
  form, 13 call sites renamed; bind-mounted over the image's headers (FlashInfer JITs the
  SSU module at each start). Equivalent here because no SSU kernel launches as a cluster.
  Results: 0 syscall refs, STACK 0, first-launch drop 0 MiB (was 1,634); error vs an fp64
  reference identical to Triton; 23-layer kernel speed within -7.8% to +4.4% of
  unpatched (single-kernel timings at batch 16 swing far more, noise).
- Kernel speed, 23 Mamba layers, us/token (MEASURED): batch 1 eager Triton 924 vs
  FlashInfer 156-170; batch 1 graph Triton 67, FlashInfer 70-112; batch 16/64 all 137-153
  (memory-bound). FlashInfer only helps eager at low batch, which is how we serve.
  The batch-1 graph numbers imply ~1.43 TB/s vs ~0.70 at batch 16, so they are probably
  flattered by L2 cache. At batch 1, `auto` picks the simple kernel (1 x 64 heads < 2 x 82
  SMs), so fi-cta's c=1 row runs no patched kernel; the patch only runs at batch >= 3.
- Flag-only alternative, no patch: `--mamba-ssu-algorithm simple` (no TMA at all).
- **Serve A/B r1 (MEASURED, 1 run each, G7a flags):** c=1 decode p50 triton 31.86,
  fi-cta 33.39, fi-simple 33.61 tok/s; c=16 per request 31.86 / 31.44 / 31.06; v1 eval
  held_out hit&grounded 48 / 45 / 43 of 90, seen_tool 13 / 15 / 8 of 25, truncated
  41 / 44 / 50 of 158; 0 errors each. By the G7a rule fi-simple's -5 is a LOSS on both
  splits in r1, pending repeats: greedy decoding is not run-to-run repeatable here
  (yesterday's triton and today's share 15/158 identical final answers; two triton runs
  differ on 20 held_out items). Triton ran first, from a cold GPU.
- **Repeats — CHOSEN before the runs, 2026-09-26 (David: "proceed with all recommended
  steps"):**
  - Two more runs per config, same script and flags (`probes/ssu_sm120/serve/run_all.sh`):
    r2 in order fi-simple, fi-cta, triton; r3 in order triton, fi-cta, fi-simple (r1 was
    triton, fi-cta, fi-simple). n = 3 per config.
  - Metrics, 3-run mean per config: held_out and seen_tool hit_and_grounded items, c=1
    decode tok/s p50, c=16 tok/s per request p50. Truncations: descriptive.
  - Quality: a config loses (wins) against triton on a split only if its mean is more
    than 4 items below (above) triton's (the G7a noise-floor rule); else tie.
  - Speed: a difference counts only if it exceeds the larger of 5% and triton's own
    min-max range across its 3 runs.
  - Adopt for `--enforce-eager` serving: fi-simple if it has no quality loss and is faster
    at c=1 (flag only, nothing to maintain); else fi-cta on the same test (needs the
    patch); else keep Triton, the default. c=16 alone never decides it (memory-bound).
  - `probes/ssu_sm120/serve/compare.py` applies this rule.
- **Repeats RESULT (MEASURED 2026-09-26; `results/ssu_sm120/serve/r{1,2,3}/`, 18/18
  steps exit 0, 0 eval errors):** 3-run means, triton / fi-cta / fi-simple: held_out
  42.7 / 44.3 / 44.0 of 90; seen_tool 11.3 / 13.3 / 11.0 of 25; c=1 decode p50 30.88 /
  32.93 / 33.46 tok/s; c=16 per request 30.17 / 31.79 / 31.73; truncated 44.3 / 47.0 /
  48.0 of 158. By the rule: quality ties everywhere (fi-simple +1.3 / -0.3 items, fi-cta
  +1.7 / +2.0); c=1 fi-simple +8.4% and fi-cta +6.7%, both past the 2.0 tok/s bar (triton's
  range); c=16 ties. **Adopted: `--mamba-backend flashinfer --mamba-ssu-algorithm simple`
  for `--enforce-eager` serving (no patch needed).** Not yet applied to any serve script.
  - r1's fi-simple -5 was noise: triton's own held_out ranged 39-48 over its 3 runs.
  - Caveats (descriptive): triton's c=1 fell each run (31.86, 30.91, 29.86) while the
    FlashInfer runs did not, so part of the margin may be drift; the win clears the bar
    by 0.6 tok/s. FlashInfer truncated ~3-4 more of 158. KV varied 1.41-1.54 GiB with
    other apps on the card.
- Upstream issue (FlashInfer / CCCL): drafted, not filed; outward-facing, needs David's go.

### G6: short training + held-out eval
- **Thinking mode — CHOSEN 2026-09-26 (David: "proceed with all recommended steps"):**
  evaluate the adapter and base Lightning in both modes; **thinking on is primary**
  (`--thinking on --max-tokens 4096`, other settings as G7a's), thinking off (G7a's
  exact settings) is secondary. Why: thinking moves base held_out by 33 points (G7a), and
  thinking off at 512 tokens truncates 48/158 v1 answers, so thinking off measures the
  cap as much as the model. The pass rule below compares like with like (same mode).
  UNKNOWN before training: whether research_dataset_v3's examples carry think blocks;
  check before the run.
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

- **Setup (2026-09-27; `probes/g6_train.py`, dry run `results/g6-dry.log`):**
  - Data: v3 as it is (David: "train on V3 as is to keep comparison to qwen like for
    like"). It carries **no reasoning text**: `thinking` is a mode flag (693 "default",
    257 "off"), and no record holds a `<think>` block.
  - Recipe: gpu-lab `training/qlora.py`'s, which trained v3 (r=16, alpha 32, dropout 0.05,
    lr 1e-4, cosine with 3% warmup, paged AdamW 8-bit, clip 1.0, 2 epochs, 8 records per
    step, max length 1024, assistant-only loss). Batch 1 x 8 with the loss normalised over
    the step's assistant tokens is the same objective as qlora.py's 2 x 4 under
    transformers 5.x's Trainer. Forced by the model: placement A, F1's configuration
    (attention bf16, lean scan, chunked CE), no fp32 upcast (v3's flag is not recorded).
  - Template: Lightning writes `<think></think>` before every assistant turn, and vLLM's
    thinking-off generation prompt ends with exactly that (checked), so "off" records mask
    it as prompt: 257 records, 402 blocks. qlora.py's Qwen-string insertion finds nothing
    to insert here. "Default" records train it, as qlora.py trained Qwen's empty block.
  - Truncation at 1024: 439 of 950 records here, 375 under Qwen3-8B's template (if v3 ran
    at qlora.py's default). Neither template passes 2048 (max 1892 / 1712).
  - Thermal guard (David, 2026-09-27): `GUARD=hw` = abort on hw_thermal or hw_power_brake,
    or at 90 C. G5's guard aborts on sw_thermal or at 87 C, and v3 ran at 87 C for 36 min.
- **Training (MEASURED, 2026-09-27; `results/g6-train.{json,log}`):** 238 steps, no abort,
  2,048.5 s (v3: 2,171 s). Loss 1.001 first step, mean 0.488 epoch 1 and 0.158 epoch 2,
  0.0025 last step. Torch peak 19.42 GiB; max 80 C and 154 W; 0 throttle flags in 4,036
  samples, so G5's guard would not have tripped either. 11,359,232 trainable parameters
  in 93 modules; chunked CE ran 1,900 times (950 x 2). Adapter `results/g6-train-adapter/`
  (not in git; checkpoints 119 and 238). The log's gpu-lab stamp (5512534) was taken at
  the end; the run started at c3e9a46, and `training/` is identical in both.

- **G6 RESULT (MEASURED, 2026-09-27; `probes/g6_eval.sh`, `probes/g6_compare.py`,
  `results/g6-compare.json`, `results/research-eval-*-lightning-{g6-think-4k,think-4k,
  g6-nothink}.*`): FAILS the rule in both modes, on one task row each; ties the Qwen3-8B
  v3 adapter like for like (1 win, 0 losses, 21 ties).**
  - Serving: one vLLM server for base and adapter (G7a's flags plus LoRA). 22 runs, 0 items
    in status `error`, 0 unparsed `<tool_call>` turns; all 1,681 calls `xml_function`.
    Harness `probes/g7a_eval.py`. The first attempt ran the bare `research_eval.py`, which
    parses neither Lightning's XML calls nor its open `<think>`: v1 scored 0 with every call
    missed. Stopped, outputs deleted; `results/g6-eval-attempt1-unpatched-harness.log`.
  - **Rule, thinking on (primary): FAIL on `rocky_task`, 0.25 vs 0.65 (-8 items).**
    `held_out` WINS, 0.889 vs 0.789 (+9); no loss on trap (+8, a win), trap_control,
    no_tool, task (-1), alert (+1) or promql (-1). Also wins: held_out2 +8, two_flag +5,
    trap2 +7, rocky_held_out +10, rocky_trap +5, general over_trigger +8. General
    correct_where_scorable -6 (not in the rule; counted over n=45, so overstated).
  - Thinking off (secondary, against G7a's base): the same rule FAILS on `task`, 0.55 vs
    0.80 (-5); rocky_task -3 ties. Wins: held_out +44 (1.000 vs 0.511), held_out2 +43,
    rocky_held_out +32, seen_tool +13, trap2 +11, trap +8, rocky_trap +8, two_flag +7,
    trap3 noticed +5.
  - **Like for like, both adapters thinking off, G6 vs the Qwen3-8B v3 adapter: 1 win,
    0 losses, 21 ties.** The reflex is learned equally (held_out 1.000 each, rocky_held_out
    1.000 each); the win is alert rules, 0.556 vs 0.000, where v3 collapsed and G6 kept
    base's level. Task rows equal (task 0.55 each; rocky_task 0.40 vs 0.25, a tie).
  - **Diagnostic, not part of the rule: with thinking on, 64 of 478 adapter answers end
    inside the open think block** (finish `stop` after 50-273 tokens, never `</think>`), so
    the harness sees no answer; base: 0 of 478. rocky_task 8 of 20; v1 17 of 158, where 14
    of the 16 flag items hold the right flag in the hidden text. Cause (inference, matching
    the template check above): training shows `<think></think>` then the answer, while the
    thinking-on prompt ends inside an open `<think>\n`, so the adapter answers there and
    stops. Most of the thinking-on rocky_task loss is this, not the model's knowledge.
  - Server check, base v1 on this LoRA-enabled server vs G7a's: thinking off, all rows tie
    (-3 to +2 items); thinking on, held_out -5 (0.789 vs 0.844), just past the 4-item
    floor, the rest tie. Thinking-on base moves ~5 items run to run.
  - Reading: like every 8B adapter, G6 wins the lookup reflex and the traps and loses a
    real-task row to its own base; unlike v3 it keeps base's alert rules. Thinking on adds
    a Lightning-specific defect that the training render causes. Open, each a new
    configuration beside G6 rather than a rerun of it: (1) render "default" records as the
    thinking-on prompt looks (`<think>\n` as prompt, then `</think>` and the answer trained);
    (2) self-generated reasoning traces (v4; David asked 2026-09-27 whether traces help a
    practical model). G7's merge path is not needed: vLLM serves the LoRA directly.

- **G6r: option (1) above, trained beside G6 (David, 2026-09-27).** Same data, recipe and
  guard as G6; only the render of "default" records changes (`probes/g6_train.py`, env
  `RENDER=think`; `RENDER=g6`, the default, reproduces G6's data stats exactly).
  - Render: each "default" assistant turn is `assistant\n<think>\n` as prompt, then
    `</think>`, the answer and `<|im_end|>` trained (986 turns); the 257 "off" records are
    G6's. **Design A (David's choice):** one sequence per record, as G6, so earlier turns in
    a record carry `<think>\n</think>` where the server sends history as `<think></think>`.
    Design B, one sequence per assistant turn (1,824 sequences, 1.92x), would remove that.
    David: re-weigh A vs B if Lightning becomes the lab's default model.
  - Parity guard (MEASURED, `results/g6r-dry.log`): per trained turn, the tokens before the
    trained span vs `apply_chat_template(history, add_generation_prompt=True,
    enable_thinking=...)`, the prompt the eval server builds. `RENDER=think` PASS: 1,388 of
    1,388 boundaries exact, 1,095 whole prompts exact, 293 turns see a history 312 tokens
    longer in all (the newline in earlier `<think>\n</think>` blocks). `RENDER=g6` FAIL:
    986 bad boundaries, every "default" turn, i.e. G6's defect caught before training.
    `RENDER=think` refuses to train on a FAIL.
  - Training (MEASURED, `results/g6r-train.{json,log}`): 238 steps, no abort, 2,066.8 s
    (G6 2,048.5 s). Loss 1.115 first step, mean 0.491 epoch 1 and 0.159 epoch 2, 0.0024
    last (G6 1.001, 0.488, 0.158, 0.0025). Torch peak 19.42 GiB, as G6; max 80 C and
    154.8 W; no hw_thermal or hw_power_brake sample (sw_power_cap, the power limit holding,
    in 4,054 of 4,080, as G6's 4,011 of 4,036). 439 records truncated at 1024, as G6.
    Adapter `results/g6r-train-adapter/` (not in git).

- **G6R RESULT (MEASURED, 2026-09-28; `LABEL=g6r probes/g6_eval.sh`, `LABEL=g6r
  probes/g6_compare.py`, `probes/g6r_analysis.py`, `results/g6r-{compare,analysis}.json`,
  `results/research-eval-*-lightning-g6r-*.*`): FAILS the rule in both modes: thinking on
  on `rocky_task` (-7 items; G6 -8), thinking off on `task` and `rocky_task`. Against G6:
  4 wins, 0 losses, 40 ties. The render fix cut the think trap from 64 turns to 24 but
  did not move the task rows: their loss is template collapse, not the trap.**
  - Eval: 14 runs, all exit 0 (~15 min; base outputs reused from G6). 956 items, 0 in
    status `error`, 938 calls all `xml_function`, 0 unparsed `<tool_call>` turns. 21 items
    hit the call limit (G6 15, base 124).
  - **Rule, thinking on (primary): FAIL on `rocky_task`, 0.30 vs 0.65 (-7).** `held_out`
    WINS, 0.956 vs 0.789 (+15); no loss on trap (+9, a win), trap_control (+1), no_tool
    (+5, a win), task (-2), alert (-3) or promql (+1).
  - Thinking off (secondary): FAILS on `task`, 0.50 vs 0.80 (-6), and `rocky_task`, 0.30
    vs 0.55 (-5; G6 -3, a tie). G6r vs G6 on thinking-off rocky_task is -2, a tie
    (INFERENCE: run-to-run movement, as base moves ~5 items, rather than a new defect).
  - **G6r vs G6:** thinking on, wins held_out (+6), seen_tool (+8) and general
    correct_where_scorable (+6); thinking off, wins promql (+5). **Alert fell, 2/9 vs 6/9
    thinking on (-4) and 3/9 vs 5/9 off**: ties by the 4-item rule only because the set
    has 9 items. Like for like, thinking off, vs the Qwen3-8B v3 adapter: 0 wins, 0
    losses, 22 ties (G6's one win, alert, is now +3, a tie).
  - **The think trap, thinking on, all seven sets:** turns that never write `</think>`,
    base 0 of 1,206, G6 64 of 911, G6r 24 of 939. Every one ends by choice (`stop`), none
    at the token cap: G6r 20-100 tokens (median 47), G6 30-351 (median 65; a per-turn
    count does not reproduce the G6 bullet's 50-273). The adapter writes its usual
    answer inside the open think block and stops, so the harness sees none. 23 of G6r's 24
    are the turn right after a tool result. None are in task or rocky_task: held_out 3,
    held_out2 6, rocky_held_out 5, the trap and trap_control sets 7, fix_cmd, alert and
    trap3 1 each. G6's 469 "Here's a thinking process" openers are gone (0).
  - **G6r does not think.** It writes `</think>` at once, with empty reasoning, on 843 of
    939 thinking-on turns (base 0 of 1,206; G6 0 of 911). That is what the render trains,
    since v3 holds no reasoning text, so with thinking on G6r acts as a thinking-off model;
    its task rows read the same in both modes (task 0.50, rocky_task 0.30).
  - **rocky_task, thinking on, per item:** G6r loses 8 of base's 13 hits and gains 1; 0
    trapped. 7 of the 8 are G6's lost items (dnf-14 recovered, xfs_repair-11 newly lost).
    Read by hand, 7 of the 8 are template answers (scontrol-1 made four lookups, two
    refused, and hit the call limit without answering): one flag from `--help` as the whole
    answer (`xfs_growfs -r`, "grow realtime section", to fill a partition; `srun -I` for an
    interactive shell; `sbatch -c` with no memory flag; `firewall-cmd
    --runtime-to-permanent` for "apply permanent rules"), or the trap template denying a
    real feature ("`scontrol` does not have a `resume` command"); squeue-3 looked up
    `sbatch --help`. Base answered each with a full procedure. v3's answers carry the
    shape: of 950 records, 544 hold an answer containing "Based on \`", 329 "Based on
    \`...--help\`" and 108 "I checked \`" (regex counts, MEASURED 2026-09-28).
  - This refutes G6 RESULT's reading that most of G6's thinking-on rocky_task loss was the
    trap: with no trapped item in the split, the loss is -7 vs -8. The pre-run best case
    (ARITHMETIC: remove the 4 trapped losses, -8 -> -4, a tie) did not happen.
  - Reading: the render did what a render can (40 fewer trapped turns, 4 wins over G6, no
    losses) and cannot reach the failing rows, which come from the data (INFERENCE,
    supported by the per-item reads). Open, David's call: task-shaped examples with full
    procedures in the data, or option (2), v4 reasoning traces, which would also give
    thinking-on mode something to do. The 24 remaining trapped turns matter only if G6r is
    served.

- **G6p: the data lever, pre-registered 2026-09-28 before training (David approved "do both
  recommended next steps"; design fixed by the lead).** G6r with v3 plus 200 task-shaped
  records whose answers are full procedures. Everything else is G6r's: `RENDER=think`,
  same recipe and guard, same eval, same pass rule.
  - Why: G6 and G6r lose rocky_task (and task, thinking off) by answering with one flag
    from `--help` or a false "does not have X" denial (G6R RESULT, read by hand). v3
    teaches that shape: 544 of its 950 answers contain "Based on \`" (MEASURED). If the
    data is the lever, records that ask for a goal and answer with a whole command should
    move the task rows; if they do not move them, the data is not the lever.
  - Context (MEASURED, `results/g6r-dry.log:8`): 439 of v3's 950 records exceed the
    1024-token cap. Per record (`probes/g6p_fit.py`, `results/g6p-fit.json`): 436 lose
    their final answer entirely, 3 keep part of it, 511 keep all of it. On 15 long-help
    tools training saw almost no answers: none at all for git branch (0 of 20), git tag
    (0/18), git fetch (0/16), docker compose (0/7), find (0/4); a few for tar (3 of 51),
    journalctl (3/48), docker run (3/46), grep (2/43), systemctl (1/40), nvidia-smi
    (3/39), git rebase (1/34), df (4/35), docker build (5/30), git commit (6/26). On each
    of those ten the kept count equals its number of lookup-failed records, whose tool
    message is a one-line error (MEASURED counts). INFERENCE from the equal counts: no
    answer that follows real help text was trained on any of the 15, so G6 and G6r learned
    the call there but not the answer. Four of them (tar, journalctl, grep, git rebase)
    are exactly the tools of the eval's seen_tool split.
  - The same cap shaped G6p's tools (MEASURED, Lightning's tokenizer, g6_train's render):
    the prompt, tools schema and one call take about 500 tokens, a 3-step answer about
    105, so only 13 of v3's tools leave room for an answer after their help text, and
    `git log -h` followed by `man git-log` would need 291 tokens more than the cap allows.
    The lead's decision (option a): also use other docker, git and cargo subcommands that
    no eval list names, and let the second call be a sibling subcommand's help instead of
    a man page; every new record must fit with its whole answer.
  - Data (MEASURED, `probes/g6p_build.py`, `results/g6p-build.json`,
    `results/research_dataset_g6p.json`): v3's 950 records unchanged (checked) plus 200
    new ones = 1,150. 170 hand-written task specs (each spec's first phrasing, plus a
    second phrasing for 30 drawn with the seed). Help captured on this host with gpu-lab's
    own `run_cmd` and cut by its `get_observation_for_flag`, as v3's was.
    - Tools: 69 answering tools: 13 of v3's (docker ps, images, exec, stop; git log,
      show, status, reset, diff; cargo clean, clippy; free; curl) and 56 other
      subcommands (27 docker, 20 git, 9 cargo). None is named by any eval list, and none
      is the tool of any eval item.
    - Shape: 157 answers are one complete command in a code block with a sentence of
      reasons, 43 are 2-3 numbered steps with a reason each. 40 of 200 records (20.0%)
      make a second call because the first help lacks the feature (the builder checks the
      needed token is absent from the first output and present in the second); none of
      those answers denies anything.
    - Openers: 13 templates, the most common on 20 records (10.0%); "Based on \`" and
      "I checked \`" on 0. Thinking: 67 "off", 133 "default" (v3's ratio, a third off).
    - Grounded: 200 of 200. Every flag on every answer line is in the captured help text
      (research_eval's `grounded_token`; stricter than the eval, which also grounds in the
      man page). No answer line names a flag only to deny it, and research_eval's denial
      pattern matches no answer.
    - Contamination (all 478 items of the seven eval sets, as the base runs asked them):
      0 new records share a tool and flag set with an eval item. Reported, not enforced:
      on the binary alone, git notes, git revert and git stash records that use only `-m`
      share that flag with seen_tool's git rebase `--merge`/`-m` item. Word-set Jaccard,
      new prompt vs eval prompt, max 0.389 (free's committed-memory prompt vs promql's
      "How much GPU memory is in use..."), under the 0.5 limit.
    - Fit: new records 553-985 tokens, median 810; 0 over 1024. Final answers 23-98
      tokens, median 49.
  - Dry run (MEASURED, `results/g6p-dry.log`): 1,150 records, 439 over the cap (all v3),
    924,522 tokens after the cap (the fit probe's total agrees), 68,992 trained tokens
    (G6r 52,110). Parity guard `RENDER=think` PASS: 1,828 of 1,828 boundaries; 453 turns
    see a longer history (design A's newline; G6r 293). Default path unchanged:
    `RENDER=think DRY_RUN=1` with no `DATASET` reproduces `results/g6r-dry.log` byte for
    byte (`results/g6p-dry-v3.log`), and `LABEL=g6r probes/g6_compare.py` rewrites
    `results/g6r-compare.json` identically.
  - Run: `DATASET=/out/research_dataset_g6p.json RENDER=think GUARD=hw probes/gpurun.sh
    g6p-train /probes/attn_bf16.py g6_train.py g6p-train`, then `LABEL=g6p
    probes/g6_eval.sh results/g6p-train-adapter` and `LABEL=g6p probes/g6_compare.py`
    (and again with `COMPARE_TO=g6r`). 144 steps per epoch, 288 in all, 9 warmup
    (ARITHMETIC); about 42 min at G6r's 8.7 s per step (ARITHMETIC, an upper bound since
    the new records are shorter than v3's median).
  - **Pass rule, copied from G6r:** WIN held_out, LOSE none of trap, trap_control,
    no_tool, task, rocky_task, alert, promql; win/loss = hit_and_grounded differs by >4
    items, else tie; thinking on primary, off secondary, like with like. (As in G6 and
    G6r, `probes/g6_compare.py` scores each split by its headline metric:
    hit_and_grounded on flag and task splits, the denial heuristic on traps, over-trigger
    and correct-where-scorable on no_tool, correct on alert and promql.)
  - Comparisons: vs base Lightning = the gate. vs G6r (`COMPARE_TO=g6r`) and vs the
    Qwen3-8B v3 adapter = informative, same verdict rule.
  - Diagnostics, fixed now: the think trap (thinking-on turns with no `</think>`, all
    seven sets; base 0 of 1,206, G6 64 of 911, G6r 24 of 939); thinking-on turns with
    empty reasoning (G6r 843 of 939); for task and rocky_task, each item base hits and
    G6p misses, classed mechanically as a false denial (research_eval's denial pattern
    matches the answer), else a single-flag answer (it cites exactly one flag), else
    other, then read by hand.
  - Training (MEASURED, `results/g6p-train.{json,log}`): 288 steps, no abort, 2,449.3 s
    (G6r 2,066.8 s for 238). Loss 1.182 first step, mean 0.515 epoch 1 and 0.200 epoch 2,
    0.061 last (G6r 1.115, 0.491, 0.159, 0.0024). Torch peak 19.42 GiB, as G6r; max 79 C
    and 153.8 W; no hw_thermal or hw_power_brake sample (sw_power_cap in 4,834 of 4,855).
    Chunked CE ran 2,300 times (1,150 x 2). Adapter `results/g6p-train-adapter/` (not in
    git).

- **G6P RESULT (MEASURED, 2026-09-28; `LABEL=g6p probes/g6_eval.sh`, `LABEL=g6p
  probes/g6_compare.py` with and without `COMPARE_TO=g6r`, `probes/g6p_analysis.py`,
  `results/g6p-{compare,compare-vs-g6r,analysis}.json`,
  `results/research-eval-*-lightning-g6p-*.*`): FAILS the rule thinking on, on a new row,
  `promql` (-7 items); `rocky_task` no longer fails (-4, a tie; G6r -7). Thinking off,
  the same rule PASSES, the first Lightning adapter to do so. Against G6r: 3 wins, 2
  losses, 39 ties. The data moved the task rows; it also moved promql the wrong way.**
  - Eval: 14 runs, all exit 0, 12 min 55 s (base outputs reused from G6). 956 items, 0 in
    status `error`, 963 calls all `xml_function`, 0 unparsed `<tool_call>` turns. 39 items
    hit the call limit (G6r 21, G6 15, base 124), 19 of them in promql.
  - **Rule, thinking on (primary): FAIL on `promql`, 0.278 vs 0.667 (-7).** `held_out`
    WINS, 0.967 vs 0.789 (+16); no loss on trap (+9, a win), trap_control (+2), no_tool
    (over-trigger +5, a win; correct +0), task (+3, 0.75 vs 0.60), rocky_task (-4, 0.45
    vs 0.65) or alert (-3, 2/9 vs 5/9, as G6r).
  - Thinking off (secondary): the rule PASSES. held_out +40 (0.956 vs 0.511), trap +8;
    task +1 (0.85 vs 0.80), rocky_task -2 (0.45 vs 0.55), promql -1, alert -3: ties.
  - **G6p vs G6r:** thinking on, wins task (+5, 0.75 vs 0.50) and held_out2 (+6), loses
    promql (-8, 0.278 vs 0.722); thinking off, wins task (+7, 0.85 vs 0.50), loses promql
    (-7, 0.278 vs 0.667); rocky_task +3 in both, a tie. Like for like, thinking off, vs the
    Qwen3-8B v3 adapter: 1 win (task, +6: 0.85 vs 0.55), 0 losses, 21 ties.
  - **promql, why (MEASURED counts, thinking on):** G6p makes 36 bash calls and 6 promql
    calls over the 18 items (G6r 9 and 17; base 1 bash, 22 promql), and 9 items hit the
    call limit (G6r 1). Asked "What is the desktop GPU's temperature right now?", it runs
    `nvidia-smi --help` and `man nvidia-smi` instead of querying Prometheus; asked whether
    the laptop is on AC, it looks up `acpi` and answers with a flag "from memory".
    INFERENCE: every one of the 200 new records answers a goal by reading bash help, and
    none uses any other tool, so bash lookup became the reflex for live questions too.
  - **The think trap, thinking on, all seven sets:** 1 of 959 turns never writes
    `</think>` (base 0 of 1,206, G6 64 of 911, G6r 24 of 939). G6p does not think at all:
    958 of 959 turns close `</think>` at once with empty reasoning (G6r 843 of 939).
  - **rocky_task, thinking on, per item:** G6p loses 8 of base's 13 hits and gains 4
    (exportfs-9, firewall-cmd-7, ipa-16, rpm-13); G6r lost 8 and gained 1. Mechanical
    classes: 3 single-flag, 5 other, 0 false denial (G6r 2, 3, 3). By hand: 4 have the
    new shape, a whole command or steps, but the wrong command (`dnf config-manager
    --list-repos`, `exportfs -a` where `-r` is needed, `firewall-cmd
    --runtime-to-permanent`, a 2-step squeue answer about a held job); 2 are v3's
    one-flag template (`srun -I`; sbatch's option list with `--memory` for `--mem` and no
    command); 2 hit the call limit (scontrol-1 and -2 look up a nonexistent `snode`). No
    answer denies a feature; G6r's "`scontrol` does not have a `resume` command" is gone.
  - task, thinking on: lost 3, gained 6 (G6r lost 5, gained 3). Thinking off: task lost 2,
    gained 3; rocky_task lost 5, gained 3. All 5 losses open with "Based on \`": 3 give
    one option as the answer, 2 a one-flag command in a code block.
  - Shape, task and rocky_task together (40 items, MEASURED by regex): thinking on,
    answers with a code block 13 (G6r 3, base 32), numbered steps 3 (G6r 0), "Based on \`"
    27 (G6r 21, base 0); thinking off, code block 16 (G6r 1), "Based on \`" 29 (G6r 29).
    Many are hybrids: "Based on \`x --help\`, the command is:" followed by a code block.
  - Reading: the data is a lever. 200 records (17% of the set) turned rocky_task from a
    loss into a tie, won task against G6r in both modes, and removed the denial answers.
    The lever is partial: v3's template still writes most task answers, now often wrapped
    around a whole command (INFERENCE: 544 v3 answers against 200). It is not free: an
    all-bash data set cost promql. Open, David's call: add live-question records that
    use the promql tool beside the new ones; rewrite or drop v3's "Based on" answers so
    the new shape is not outnumbered; or option (2), v4 reasoning traces (G6p, like G6r,
    leaves thinking-on mode empty).

- **G6P2048: G6p at the length Qwen v3 was trained at, pre-registered 2026-09-28 before
  training (David: "proceed with your best judgement, i'm putting you in control"; design
  fixed by the lead).** G6p with one change, `MAX_LEN=2048`. Same data
  (`results/research_dataset_g6p.json`), `RENDER=think`, recipe, guard, eval and pass rule.
  - Why: G6's recipe line above ("max length 1024, qlora.py's default") was wrong about
    v3. The command that trained the Qwen3-8B v3 adapter (MEASURED, Claude session
    09256598, tool call at 2026-09-24T03:58Z, the only v3 training call there) was
    `qlora.py --model Qwen/Qwen3-8B --out /adapters/qwen3-8b-research-v3 --dataset
    /training/research_dataset_v3.json --max-len 2048 --save-strategy epoch`. No v3 record
    passes 2048 under Qwen's template (max 1712, above), so Qwen trained every answer; G6,
    G6r and G6p never trained 436 of them (G6p's context note). Every "like for like vs
    the Qwen3-8B v3 adapter" tally so far carries that handicap. It also clears lever (b)'s
    blocker: G6p's new records are 553-985 tokens before any reasoning text is added.
  - Change: `probes/g6_train.py` reads `MAX_LEN` (unset = 1024, as G6, G6r and G6p ran);
    `probes/gpurun.sh` passes it into the container.
  - Dry run (MEASURED, `results/g6p2048-dry.log`): 1,150 records, 0 over the cap (max
    1,892 tokens, p90 1,507), 1,092,595 tokens, 91,823 trained tokens (G6p 68,992, +33%).
    Parity guard `RENDER=think` PASS: 2,264 of 2,264 boundaries; 748 turns see a longer
    history (design A's newline; G6p 453). Default path unchanged: the same dry run
    without `MAX_LEN` (`results/g6p-dry-recheck.log`) equals `results/g6p-dry.log` except
    the one `attn_bf16 {...}` line, which that run printed because it ran through the
    wrapper and the recheck did not.
  - Memory: F1 measured 22.09 GiB device peak at seq 2048 against the 23.39 line; the
    longest record here is 1,892. Time: G6p trained 2 x 924,522 tokens in 2,449 s (755
    tokens/s), so 2 x 1,092,595 takes about 48 min (ARITHMETIC).
  - Run: `DATASET=/out/research_dataset_g6p.json RENDER=think GUARD=hw MAX_LEN=2048
    probes/gpurun.sh g6p2048-train /probes/attn_bf16.py g6_train.py g6p2048-train`, then
    `LABEL=g6p2048 probes/g6_eval.sh results/g6p2048-train-adapter`, `LABEL=g6p2048
    probes/g6_compare.py` (vs base, the gate) and again with `COMPARE_TO=g6p`.
  - **Pass rule, copied from G6p:** WIN held_out, LOSE none of trap, trap_control,
    no_tool, task, rocky_task, alert, promql; win/loss = hit_and_grounded differs by >4
    items, else tie; thinking on primary, off secondary, like with like.
  - Expectations (INFERENCE, written before the run so they can be wrong): the 436 newly
    trained answers are mostly v3's one-flag "Based on \`" shape, so task rows may lose
    some of G6p's gain; seen_tool (tar, journalctl, grep, git rebase) may gain, since
    those tools' answers are trained for the first time; promql should not move, since
    its cause (bash-only new records) is unchanged.
  - What it decides: no worse than G6p on the rule (thinking on and off) -> 2048 becomes
    the cap for lever (b). Worse on task rows -> v3's template answers are the harm, and
    rewriting them comes before lever (b).
- **G6P2048 RESULT (MEASURED, 2026-09-28; `results/g6p2048-{train,compare,
  compare-vs-g6p}.json`, `results/research-eval-*-lightning-g6p2048-*.*`): same verdict as
  G6p. Thinking on FAILS, on promql (0.333 vs 0.667, -6) and alert (0/9 vs 5/9, -5);
  thinking off PASSES. Against G6p: every row a tie in both modes. Like for like and now
  at equal length, thinking off, vs the Qwen3-8B v3 adapter: 1 win (rocky_task, 0.60 vs
  0.25), 0 losses, 21 ties.**
  - Train: 288/288 steps, 2,796 s, no guard abort, exit 0; loss 1.485 -> 0.058. Eval: 14
    runs, exit 0. (Exclude the CHECK row, base v1 G6 server vs G7a, when tallying.)
  - Thinking on: task 0.80 (G6p 0.75), rocky_task 0.60 (G6p 0.45), both ties vs base.
    Alert 0/9 vs G6p's 2/9 is 2 items, a tie with G6p, but crosses the >4 line vs base.
  - Expectations: task rows did not lose (the INFERENCE above was wrong); promql did not
    move (+1 vs G6p), as expected. Per the decision above, 2048 is the cap for lever (b).
  - fable-judge (2026-09-29): VERIFIED. `git archive 7152188 probes results` into a scratch
    directory, both compare commands rerun there: `g6p2048-compare.json` and
    `g6p2048-compare-vs-g6p.json` byte-identical to the committed ones; every number above
    matches the printout; the vs-G6p blocks have no win or loss row in either mode.
  - **Per item, thinking on (MEASURED, `probes/tool_choice_items.py`,
    `results/g6p2048-tool-choice.json`): the loss is tool choice, not knowledge.**
    - Alert: base answers all 9 with no tool call (5 correct). G6p and G6P2048 call bash on
      every item (27 calls each, `man prometheus-alerting-rules`, `man prometheus`, ...),
      hit the 3-call limit on 7 and 6 items, and answer 2 and 3.
    - promql: base calls promql 22 times and bash once. G6P2048 calls bash 33 times and
      promql 7; the 7 items where it called promql are 6 correct, the 11 where it called
      bash are all wrong (`nvidia-smi --help` for a Prometheus question). G6p: the same.
    - Every adapter turn on both sets opens with an empty `</think>` (alert 36 of 36,
      promql 58 of 58; base 0): neither adapter reasons before choosing a tool.
    - Training never showed the promql tool: `g6_train.py` rendered every record with
      TOOLS (bash, web_search) only, while the eval's promql set adds a third tool.

- **G6q: G6P2048 plus tool-choice records, pre-registered 2026-09-29 before training
  (David: "you are go to proceed with all next steps"; design fixed by the lead).** G6p's
  1,150 records unchanged plus 98 new ones, `MAX_LEN=2048`; `RENDER=think`, recipe, guard,
  eval and pass rule as G6P2048.
  - Why: G6P2048's per-item read above. Both failing rows lose on tool choice: bash for
    live questions, bash lookups instead of an answer for rule writing. All 200 of G6p's
    records answer via bash help, and no record ever showed the promql tool.
  - Change to the trainer: `probes/g6_train.py`'s `tools_for(record)` renders `TOOLS` plus
    the record's `extra_tools`, in the encode and in the parity guard. Default path
    unchanged (MEASURED): the G6P2048 dry run rerun with this code,
    `results/g6q-defaultpath-dry.log`, equals `results/g6p2048-dry.log` byte for byte.
  - Data (MEASURED, `probes/g6q_build.py`, `results/g6q-build.json`,
    `results/research_dataset_g6q.json`):
    - 58 promql records (29 specs x 2 phrasings; 4 make two calls). Each carries the
      promql tool in `extra_tools` with the description the eval's promqlcat set builds
      (metric catalog appended, read from the same Prometheus). Tool outputs are real:
      research_eval's own `execute()` against lab-desktop:9090 at build time; answers are
      computed from the returned values. Metrics: CPU temperatures, GPU clocks, battery
      health/cycles/discharging/power, power-limit default and max, throttle reasons,
      scrape_ok, memory-controller utilization, scrape duration and samples, llama-swap
      load/RAM/swap; 3 "no data" answers (empty `vllm:` counters, `node_load1`), each
      matching research_eval's NODATA pattern.
    - 40 alert records (20 specs x 2 phrasings), answered with no tool call: an opener
      (6, none on more than 7), a `groups:` YAML block and one sentence. Every rule scores
      valid and correct under research_eval's own `score_alert` (promtool check and unit
      tests in prom/prometheus:v3.14.0) on series written for it: 20 of 20.
    - Thinking: 33 "off", 65 "default", seeded (G6p's third).
    - Contamination: no promql query names a metric that any promql eval truth query
      names (14 excluded, `up` among them); no rule names a metric any alert eval item
      names (10 excluded) or shares an alertname; word-set Jaccard vs all 478 eval
      prompts max 0.467 (limit 0.5; four phrasings were rewritten to get under it).
    - Not reproducible byte for byte: a rebuild reads Prometheus again, so values move.
      The committed dataset is the record.
  - Dry run (MEASURED, `results/g6q-dry.log`): 1,248 records, 0 over 2,048 (max 1,892,
    p90 1,503), 1,178,797 tokens, 99,387 trained tokens (G6P2048 91,823). Parity guard
    `RENDER=think` PASS, 2,424 of 2,424 boundaries (G6P2048 2,264; +160 = 54 x 2 + 4 x 3 +
    40, ARITHMETIC); 785 turns see a longer history (design A's newline; G6P2048 748).
  - Time: 156 steps per epoch, 312 in all, warmup 9; about 50 min at G6P2048's 781
    tokens/s (ARITHMETIC).
  - Run: `DATASET=/out/research_dataset_g6q.json RENDER=think GUARD=hw MAX_LEN=2048
    probes/gpurun.sh g6q-train /probes/attn_bf16.py g6_train.py g6q-train`, then
    `LABEL=g6q probes/g6_eval.sh results/g6q-train-adapter`, `LABEL=g6q
    probes/g6_compare.py` (vs base, the gate) and again with `COMPARE_TO=g6p2048`; then
    `probes/tool_choice_items.py results/g6q-tool-choice.json base g6p2048 g6q`.
  - **Pass rule, unchanged:** WIN held_out, LOSE none of trap, trap_control, no_tool,
    task, rocky_task, alert, promql; win/loss = the headline metric differs by >4 items,
    else tie; thinking on primary, off secondary, like with like.
  - Expectations (INFERENCE, written before the run so they can be wrong): promql stops
    losing, since every training record that shows the promql tool uses it (a cue the
    eval also carries); alert is less sure, since its prompt has no such cue and 1,150
    bash records say "look it up first"; no other row moves against G6P2048. Diagnostics:
    promql calls vs bash calls on promqlcat, zero-call alert items, empty-think turns.
  - What it decides: passes thinking on -> the first Lightning adapter to pass the gate in
    both modes, and the base for lever (b). promql fixed, alert not -> the schema cue is
    the lever and alert needs more records or a cue; neither -> a tool-choice share of 8%
    is not enough, and lever (b) (reasoning before the call) is next.
- **G6Q RESULT (MEASURED, 2026-09-29; `results/g6q-{train,compare,compare-vs-g6p2048,
  tool-choice}.json`, `results/research-eval-*-lightning-g6q-*.*`): PASSES the rule in
  both modes, the first Lightning adapter to pass thinking on. Thinking on: promql 0.833
  vs base 0.667 (+3), alert 7/9 vs 5/9 (+2), both ties now instead of losses; held_out
  +18 wins; no row loses. vs G6P2048: promql +9 and alert +7 thinking on (+10, +6 off),
  all other rows ties. Like for like, thinking off, vs the Qwen3-8B v3 adapter: 2 wins
  (promql 0.833 vs 0.500, alert 7/9 vs 0/9), 0 losses, 20 ties.**
  - Train: 312/312 steps, 2,970 s, no guard abort, exit 0; loss 1.713 first step, mean
    0.474 epoch 1 and 0.170 epoch 2, 0.074 last. Torch peak 19.86 GiB; max 77 C and
    151.6 W; sw_power_cap only. Parity in the run: 2,424 of 2,424. Adapter
    `results/g6q-train-adapter/` (not in git).
  - Eval: 14 runs, exit 0. 956 items, 0 in status `error`; 886 calls, all `xml_function`;
    9 items hit the call limit (G6p 39).
  - Tool choice, thinking on (`probes/tool_choice_items.py`): promqlcat 19 promql calls,
    0 bash, 18 of 18 answered (G6P2048: 33 bash, 7 promql, 9 at the call limit). Alert: 6
    of 9 answered with no call, all 6 correct; 3 still look things up (bash 7 calls,
    web_search 2), of which ScrapeFlapping passes and the two with two metrics
    (BatteryLowOnBattery, DataDiskAlmostFull) do not.
  - Expectations: promql, as predicted; alert did better than predicted (INFERENCE was
    "less sure").
  - Read by hand: alert answers are rules for the eval's own metrics in the trained shape
    (opener, `groups:` YAML, one sentence), e.g. TargetDown `up == 0` for 2m. ScrapeFlapping
    is correct, but its text claims a basis in "`promtool rule --help` output" that no
    call returned. The three promql misses are new errors: `up{job="scrape"} == 0` and
    `up{job="gpulab"}`, both invented job labels, empty results reported as "none down" and
    "none up"; and the desktop's VRAM given as 33 MiB when GiB was asked. INFERENCE: the
    3 "no data" records taught "empty -> none" without teaching a label check first.
  - rocky_task drifts: 0.45 thinking on (G6P2048 0.60, base 0.65; -4 vs base, one item
    from a loss), 0.40 off (-3). The 4 items lost vs G6P2048 (dnf-14, firewall-cmd-7,
    sbatch-5, squeue-4) are all v3's "Based on \`x --help\`, the option is" template,
    the known template collapse, not tool choice.
  - Think: 4 of 919 thinking-on turns never close `</think>` (rocky_trap-sinfo,
    held_out2-rsync-2, trap2-strace, two_flag-strace; G6p 1); 915 of 919 have empty
    reasoning. G6q still does not think, so lever (b) still has its target.
  - What it decides (per the pre-registration): G6q is the base for lever (b), at 2048.
    Merging to main is David's call. Open, in order: the template collapse on rocky_task
    (v3's 464 "Based on" answers, the largest remaining risk: one more item is a loss);
    empty reasoning (lever b); label checking before trusting an empty promql result.

- **G6t: lever (b), base Lightning's own reasoning traces on G6q, pre-registered
  2026-09-29 before training (David: "you are go to proceed with all next steps"; design
  fixed by the lead).** G6q's data and recipe (`MAX_LEN=2048`, guard, eval, pass rule);
  131 of G6q's "default" records have their thinking-on turns replaced by base Lightning's
  own completions, reasoning included, where the eval's own scorer accepts the answer.
  - Why: G6q passes but writes empty reasoning on 915 of 919 thinking-on turns. Base
    reasons on every turn and still wins rocky_task (0.65 vs 0.45). The question: does
    training base's own reasoning break the empty-think reflex, and what does it cost?
  - Collection (MEASURED, `probes/g6t_collect.{py,sh}`, `results/g6t-traces.jsonl`,
    `results/g6t-collect.log`): base Lightning NVFP4 served as `probes/g6_eval.sh` does,
    minus LoRA. Each of the 605 "default" records of eight mechanically checkable types
    ran through research_eval's own `run_item` with g7a_eval's two patches, thinking on,
    G6q's tools (promql records carry `extra_tools`), 3 calls, window 4000, real tools;
    a wrapper on research_eval's `execute` records each tool output so the history is
    rebuilt exactly as `run_item` sent it (every call re-parses from its turn's text).
    Scored by research_eval's own `score()` on an eval-shaped item per record type
    (cli_grounded -> held_out, compose -> two_flag, task_procedure -> task, both trap
    types -> trap, alert_direct -> alert with G6q's series, arithmetic -> no_tool);
    promql_live: promql calls only and every number of G6q's reference answer, recomputed
    from Prometheus, within 5% or 1. Accepted only if answered, every turn `stop` with a
    closed, non-empty `</think>`, <= 3 calls, no claimed unrun lookup, the tool looked
    up (flag and trap kinds), and prompt + completion <= 2,047 tokens per turn. Attempt
    0 greedy, as the eval; attempts 1-3 sample (0.6, top-p 0.95), until one is accepted.
    1,779 attempts in 3,108 s; 266 of 605 records accepted.
  - Pilot first (`results/g6t-pilot.{jsonl,log}`, 40 records, greedy): 11 accepted. Read
    by hand: rejections are base's real habits under the eval's rules, not a harness
    fault: it opens with web_search (unavailable), tries `--help | grep` pipes (refused)
    and spends its 3 calls (base: 124 call-limit items in G6's eval), or answers from
    memory without looking the tool up.
  - **Clean traces only (the lead's decision):** of the 266, 135 carry 112 refused calls
    and 69 web_search calls. Training them would import the call-spending habit that is
    base's main eval failure and not G6q's (9 call-limit items). `probes/g6t_build.py`
    attaches only traces whose every call executed: 131 records, 216 turns (arithmetic
    30, promql 18, alert 16, cli_grounded 60, compose 5, trap_refusal 1, asserted_trap 1,
    task_procedure 0); 46 answer with no call, 85 with one (67 bash, 18 promql).
    Reasoning 26-2,134 characters, median 311. Output `results/research_dataset_g6t.json`,
    audit `results/g6t-build.json`; G6q's fields unchanged on every record (checked).
  - Change to the trainer: `RENDER=trace` in `probes/g6_train.py`. A record with a trace
    becomes one sequence per turn (design B, forced: the eval sends history without
    reasoning, while Lightning's template keeps reasoning in history after the last user
    message, so one sequence per record would show later turns a history the server
    never sends). Prompt: `apply_chat_template(history, tools, add_generation_prompt,
    enable_thinking=True)`, masked; trained: base's completion text + `<|im_end|>`.
    Other records encode exactly as `RENDER=think`. Guard: every trace prompt must have
    the token count vLLM's `/tokenize` gave that turn at collection.
  - Dry run (MEASURED, `results/g6t-dry.log`): 1,333 sequences (1,117 records + 216 trace
    turns), 0 over 2,048 (max 2,045), 1,276,477 tokens, 143,335 trained (G6q 99,387).
    Trace guard PASS, 216 of 216 prompt counts equal (completion counts equal 212 of 216:
    re-tokenizing the text differs on 4); `RENDER=think` parity on the other records PASS.
    Default path unchanged: `RENDER=think` on G6q's data with this code
    (`results/g6t-defaultpath-dry.log`) equals `results/g6q-dry.log` byte for byte.
  - Time: 167 steps per epoch, 334 in all, warmup 10; about 55 min at 781 tokens/s
    (ARITHMETIC).
  - Run: `DATASET=/out/research_dataset_g6t.json RENDER=trace GUARD=hw MAX_LEN=2048
    probes/gpurun.sh g6t-train /probes/attn_bf16.py g6_train.py g6t-train`, then
    `LABEL=g6t probes/g6_eval.sh results/g6t-train-adapter`, `LABEL=g6t probes/g6_compare.py`
    (the gate) and with `COMPARE_TO=g6q`; `probes/tool_choice_items.py
    results/g6t-tool-choice.json base g6q g6t`; empty-reasoning and think-trap counts on
    all thinking-on turns.
  - **Pass rule, unchanged** (G6q's). Primary question beside it: thinking-on turns with
    non-empty reasoning, G6q 4 of 919.
  - Expectations (INFERENCE, written before the run so they can be wrong): reasoning
    appears on some thinking-on turns, most on arithmetic-, alert- and promql-like items,
    little on task rows, since 216 reasoning turns sit beside about 1,300 empty ones;
    answers get longer and more Markdown-heavy (base's style); rows mostly tie G6q. Risks:
    longer turns reach the call limit or 4,096-token cap more often; the think trap
    (no `</think>`) could return.
  - What it decides: reasoning on task rows and rocky_task up -> scale clean traces
    (more attempts, task records). Reasoning appears only on the traced kinds -> the
    share is too small or the reflex is per-kind; rocky_task no better -> base's reasoning
    alone is not the missing piece there, and the template collapse (v3's "Based on"
    answers) is next. A new loss vs base -> G6t is dropped, G6q stays the candidate.
- **G6T RESULT (MEASURED, 2026-09-29; `results/g6t-{train,compare,compare-vs-g6q,
  tool-choice}.json`, `results/research-eval-*-lightning-g6t-*.*`): PASSES the rule in
  both modes, and it reasons: 447 of 970 thinking-on turns carry non-empty reasoning
  (median 261 characters; G6q 0 of 919). Thinking on, task WINS vs base (0.85 vs 0.60,
  +5) and rocky_task reaches base (0.65 vs 0.65; G6q 0.45). vs G6q: every row a tie in
  both modes. vs the Qwen3-8B v3 adapter, thinking off: 3 wins (rocky_task 0.55 vs 0.25,
  promql 0.778 vs 0.500, alert 5/9 vs 0/9), 0 losses, 19 ties.**
  - Train: 334/334 steps, 3,189 s, no guard abort, exit 0; loss 1.344 first step, mean
    0.472 epoch 1 and 0.208 epoch 2 (G6q 0.474, 0.170), 0.036 last. Torch peak 19.94
    GiB; max 78 C and 153.0 W; sw_power_cap only. Trace guard and parity PASS in the run.
    Adapter `results/g6t-train-adapter/` (not in git).
  - Eval: 14 runs, exit 0. 956 items, 0 turn errors; 952 calls, all `xml_function`, 0
    unparsed; 9 at the call limit (G6q 9), 2 truncated, 8 think-trapped (G6q 4).
  - Reasoning by set, thinking on (turns with reasoning): v1 151/309, v2 139/283, rocky
    105/235, promqlcat 20/39, general 19/45, trap3 10/30, alert 3/29. The expectation
    ("little on task rows") was wrong: reasoning spread to every set, though almost no
    task trace was trained.
  - vs G6q, thinking on (all ties): rocky_task +4, task +3, alert +1; trap2 -4 (0.583 vs
    0.917), held_out -3, rocky_trap -2, held_out2 -2, rocky_held_out -2, promql -2.
  - Read by hand: the 6 rocky_task items gained vs G6q (dnf-14, firewall-cmd-7, -8,
    sbatch-5, squeue-4, srun-6) include all 4 that G6q lost to v3's "Based on \`x
    --help\`, the option is" template; the reasoning now picks the whole command (`dnf
    repolist`). The 4 trap2 items lost vs G6q reach the right conclusion and fail on
    mechanics: 2 at the call limit (chronyc, rsync), 1 denial written inside an
    unclosed `<think>` (nfsstat), 1 scorer artifact (lsblk: "does **not** include any
    `--health`", but the heuristic judges only the first sentence naming the flag,
    which here says it checked). Twice the reasoning opens with v3's lookup-failed
    template ("I couldn't check the help...") although the help ran.
  - Tool choice: promqlcat 19 promql calls, 2 web_search, 0 bash (13/18 correct). Alert
    8/9 correct, all 9 answered, but every item now looks something up first (20 bash
    calls, 0 zero-call answers; G6q 6 zero-call): G6q's direct answering partly reverted.
  - What it decides (per the pre-registration): reasoning reached the task rows and
    rocky_task rose, so the next step is to scale clean traces, task records above all.
    Base gave 0 clean task traces (it spends its calls), so the generator for more should
    be one that looks up efficiently; G6t itself is the candidate (INFERENCE). Both G6q
    and G6t pass; G6t is the stronger on task rows, G6q on traps and promql. Merging
    either to main is David's call.

- **G6u: G6t with more traces, task records above all, pre-registered 2026-09-29 before
  training (design fixed by the lead, under David's "proceed with all next steps").**
  G6t's data and recipe; more records carry reasoning turns, generated by G6t itself.
  - Collection round 2 (MEASURED, `results/g6u-traces.jsonl`, `results/g6u-collect.log`):
    the 474 eligible records without a clean base trace, generator G6t served as
    `probes/g6_eval.sh` serves it (`ADAPTER=... LORA=g6t probes/g6t_collect.sh ... --model
    g6t --attempts 4 --clean --skip-traced results/research_dataset_g6t.json`). New
    rejections: `--clean` (a call not executed), and reasoning that says the docs could
    not be checked though every call ran (G6t leaked v3's template that way; 0 of base's
    216 accepted turns match). 1,881 attempts, 6,034 s: **12 accepted**. The G6T RESULT
    inference that G6t could scale the traces is refuted as posed: it closes `</think>`
    empty on about half its turns, and "no reasoning" rejected 2,632 turns.
  - **Turn-level rule (the lead's decision, made on collection data, before any G6u
    training or eval):** with one sequence per turn, a turn can be left untrained without
    changing any other turn's prompt (history drops reasoning either way). A record with
    no clean accepted trace takes its first clean attempt whose only rejection is "no
    reasoning"; its trace carries `train_turns`, the turns with reasoning, and only those
    are trained (`probes/g6t_build.py PARTIAL=1`; `probes/g6_train.py` skips the other
    turns; no key = every turn, and G6t's dataset and dry run reproduce byte for byte).
  - Data (MEASURED, `results/research_dataset_g6u.json`, `results/g6u-build.json`): 403
    records carry a trace: 131 base (G6t's), 12 G6t whole, 260 G6t partial
    (cli_grounded 155, task_procedure 45, compose 36, trap_refusal 14, promql 6,
    asserted_trap 4); 502 trained reasoning turns (G6t 216). task_procedure now 48 traced
    records (G6t 0).
  - Dry run (MEASURED, `results/g6u-dry.log`): 1,347 sequences (845 records + 502 trace
    turns), 0 over 2,048 (max 2,047), 1,373,015 tokens, 223,029 trained (G6t 143,335).
    Trace guard PASS 502 of 502 (completion counts equal 496); `RENDER=think` parity on
    the other records PASS. G6t's trace path with this code (`results/g6u-g6tpath-dry.log`)
    equals `results/g6t-dry.log` byte for byte.
  - Time: 169 steps per epoch, 338 in all, warmup 10; about 57 min at G6t's 801
    tokens/s (ARITHMETIC).
  - Run: `DATASET=/out/research_dataset_g6u.json RENDER=trace GUARD=hw MAX_LEN=2048
    probes/gpurun.sh g6u-train /probes/attn_bf16.py g6_train.py g6u-train`; eval, compare
    (gate, and `COMPARE_TO=g6t`), `probes/tool_choice_items.py` and
    `probes/think_counts.py` as G6t.
  - **Pass rule, unchanged.** Primary questions beside it: reasoning share of thinking-on
    turns (G6t 447 of 970) and task / rocky_task (G6t 0.85 / 0.65 thinking on).
  - Expectations (INFERENCE, written before the run so they can be wrong): more turns
    reason, most on v2 and rocky; task and rocky_task hold or rise; trap rows stay below
    G6q, since G6t's own traps fail on call limits and only 18 trap records are traced.
    Risk: G6t's own habits (the lookup-first alert answers, "Based on" openers) are now in
    the data with G6t's reasoning.
  - What it decides: rocky_task WINS vs base (> +4) -> G6u is the candidate and more
    task traces are the lever; ties again -> the trace lever has given what it gives at
    this scale, and the next variable is the template (rewrite v3's "Based on" answers);
    any new loss vs base -> G6u is dropped.
- **G6U RESULT (MEASURED, 2026-09-29; `results/g6u-{train,compare,compare-vs-g6t,
  tool-choice,think-counts}.json`, `results/research-eval-*-lightning-g6u-*.*`): PASSES
  both modes. Thinking on vs base: task WINS 0.95 vs 0.60 (+7), trap3 noticed WINS 0.75
  vs 0.25 (+6, new), promql 0.889 (16/18, the best of any run; +4, a tie), rocky_task
  0.65 vs 0.65 (a tie again). Reasoning on 555 of 994 thinking-on turns (G6t 447 of
  970); promqlcat 37 of 39. vs G6t: every row a tie in both modes. vs the Qwen3-8B v3
  adapter, thinking off: 3 wins, 0 losses, 19 ties.**
  - Train: 338/338 steps, 3,338 s, no guard abort, exit 0; loss 0.597 first step, mean
    0.436 epoch 1 and 0.214 epoch 2, 0.066 last. Torch peak 19.94 GiB; max 79 C, 152.8 W;
    sw_power_cap only. Trace guard and parity PASS in the run. Adapter
    `results/g6u-train-adapter/` (not in git).
  - Eval: 14 runs, exit 0; 956 items, 0 turn errors, 981 calls all `xml_function`, 0
    unparsed; 12 at the call limit, 6 think-trapped, 2 truncated.
  - vs G6t, thinking on (ties): trap3 noticed +4, promql +3, task +2, rocky_held_out +2;
    rocky_trap -3, alert -2 (6/9; answers now mostly start with a web_search, 6 calls).
  - **By the rule, a tie means the next variable is the template. The per-item read
    refutes that rule's premise (MEASURED):** none of G6u's 7 rocky_task misses is a
    "Based on" answer. 3 hit the call limit (exportfs-10, firewall-cmd-7, rpm-13); 4 name
    the right command and add a flag the docs do not have (scontrol-1 `--dry-run`,
    squeue-4 and xfs_repair-11 `-N`, ipa-16 four). Base misses 7 too, 4 of them the same
    items, by the same two failures (call limit; ipa-16 with 13 invented flags). 19 of
    the 20 items are solved by at least one of base, G6q, G6t, G6u; only rpm-13 by none.
    Rewriting v3's template would touch none of the 7, so it is not run.
  - Reading: the traces carried base's reasoning and with it base's failure modes;
    rocky_task sits at base's level. The item sets that pass differ from run to run
    more than the scores do, and G6's server check found thinking-on base moving ~5
    items between identical runs. Every adapter-vs-adapter comparison since G6q is "all
    ties", so the eval cannot yet rank G6q, G6t and G6u. Next: measure that noise.

- **N1: run-to-run noise of the thinking-on eval, pre-registered 2026-09-29 before any
  repeat ran (design fixed by the lead).** No training. The contested thinking-on sets
  (v2, rocky, promqlcat, alert, trap3) run twice more for base, G6q, G6t and G6u, on a
  server started exactly as `probes/g6_eval.sh` starts it (one LoRA; base runs on G6q's
  server, as base ran on G6's), same harness settings (temperature 0, concurrency 16,
  4,096 tokens, 3 calls). Repeat r1 is the gate's own run.
  - Run: `LABEL=g6q probes/noise_eval.sh results/g6q-train-adapter "r2 r3" base`, then
    the same for g6t and g6u without `base`; outputs in `results/noise/`.
  - Summary: `probes/noise_summary.py results/n1-summary.json "r2 r3" base g6q g6t g6u`:
    per model and row, items per repeat, mean, range, and items with the same outcome in
    all three repeats; per model pair, mean difference against the larger range.
  - What it decides: (1) the spread each row has with nothing changed, so later
    decisions can be read against it; (2) which of G6q, G6t, G6u differ beyond it, on
    which rows. The candidate for main is the adapter with no row below base beyond the
    spread and the best means on the contested rows (task, rocky_task, trap2, rocky_trap,
    promql, alert, trap3); if no pair differs beyond the spread, the simplest (G6q, no
    trace machinery) is the candidate. Merging stays David's call.
  - Expectation (INFERENCE): ranges of 1-3 items on 12-20-item rows, larger on rows
    where answers hit the call limit; base the noisiest (124 call-limit items in G6).
- **N1 RESULT (MEASURED, 2026-09-29; `results/noise/`, 40 runs all exit 0;
  `results/n1-summary.json`, `results/n1-run.log`): G6u is the candidate under the
  pre-registered rule: the only adapter with no row below base beyond the spread. Two
  gate readings change once repeats are averaged: G6q's rocky_task is 4.67 items below
  base (mean 8.00 vs 12.67, range 2), past the >4 line its single run sat on (-4); and
  G6q's alert 7/9 was a high draw (7, 3, 5; mean 5.00, base 6.00).**
  - Spread with nothing changed (range over 3 repeats, items): mostly 0-3 per row;
    rocky_held_out 6 (base) and 7 (G6u); G6u rocky_task 5 (13, 15, 10); G6q alert 4.
    Items with the same outcome in all 3 repeats: G6q the steadiest (task 19/20,
    rocky_task 17/20); G6t and G6u far less (rocky_task 8/20 each): reasoning adds
    run-to-run variation.
  - Means, thinking on (base / G6q / G6t / G6u): task 13.00 / 14.67 / 16.67 / 17.67;
    rocky_task 12.67 / 8.00 / 13.33 / 12.67; trap2 3.00 / 11.00 / 8.33 / 9.00; rocky_trap
    1.00 / 10.67 / 7.67 / 6.00; promql 12.00 / 13.67 / 12.67 / 15.33; alert 6.00 / 5.00 /
    7.00 / 7.00; trap3 noticed 3.67 / 5.00 / 4.33 / 8.00; trap3 fabricated 0 / 0 / 1.00 /
    2.00.
  - Rows below base beyond the spread: G6q rocky_task (-4.67, range 2); G6t
    rocky_trap_control (+1 wrong denial, range 0); G6u none (trap3 fabricated +2 is
    within its range 2).
  - Between adapters beyond the spread: G6t and G6u beat G6q on rocky_task (G6t +5.33)
    and task (G6u +3.00); G6q beats both on the trap rows (rocky_trap +3.00 vs G6t, +4.67
    vs G6u) and held_out2 (+2.67, +3.67); G6u beats G6t on promql (+2.66) and trap3
    noticed (+3.67).
  - Why G6u's traps trail G6q's (3 repeats pooled, failures by kind): rocky_trap G6u 12
    at the call limit, 5 think-trapped, 1 answered without denying (G6q: 0, 1, 3); trap2
    G6u 3, 5, 1 (G6q 0, 3, 0). G6u keeps looking for a flag that is not there until the
    calls run out, or writes the denial inside an unclosed `<think>`; it rarely answers
    wrongly. trap3: G6u notices more (24 vs 15 of 36) and fabricates 6.
  - Reading: the reasoning traces bought task and rocky_task and cost the traps through
    call budget, not knowledge. G6u is the candidate for main (David's call). Its trap
    weakness has a concrete lever: traces that deny after one lookup.
- **G6v (not trained): trap traces from G6u, collected 2026-09-29, a negative result
  (MEASURED; `results/g6v-traces.jsonl`, `results/g6v-collect.log`,
  `results/g6v-build.json`).** The 53 "default" trap_refusal and asserted_trap records
  with no trace, generator G6u (`ADAPTER=results/g6u-train-adapter LORA=g6u
  probes/g6t_collect.sh ... --model g6u --attempts 8 --clean --max-trace-calls 1 --types
  trap_refusal,asserted_trap --skip-traced results/research_dataset_g6u.json`), 424
  attempts, 2,546 s: **0 accepted.** 368 attempts made more than one call, 368 had a
  call not executed (338 web_search, 132 refused); the habit is to look the tool up, not
  find the flag, then web_search to double-check. Allowing two calls adds none. 21
  records have a clean one-lookup denial whose only fault is an empty-reasoning lookup
  turn; under the partial rule they would add 21 trained turns (the denial turn, e.g.
  "the output does not list `--show-diff-inline-always`, so it isn't a standard git log
  option"), 1.6% of the sequences, below what N1 shows the eval can resolve. Not
  trained; the dataset (`LABEL=g6v PARTIAL=1 probes/g6t_build.py` over the three trace
  files) is not kept. Self-generated traces cannot fix G6u's traps at this scale; the
  habit to break is web_search after a lookup that does not show the flag.
- **G6w: web_search fails in the training data, pre-registered 2026-09-29 before training
  (David: "proceed with web_research training"; G6u merged to main fa7a944 the same day).**
  G6u's data and recipe; one variable: the 60 web_research records.
  - Why (MEASURED, `probes/trap_web_calls.py`, `results/g6w-web-baseline.json`, N1's three
    thinking-on repeats pooled): G6u calls web_search on 167 of 983 items and 72 of 123
    trap items; it fails 27 of those 72 against 13 of the 51 without a search. After a
    search, the failures are truncated_in_think 11 (the denial written inside the unclosed
    think, e.g. "I checked `sed --help` and `sed` has no `--atomic` option"), trap3
    fabricated 6, call_limit 5, answered without denying 5. G6t: 56 trap items searched,
    14 truncated_in_think after. G6q, same web_research records and no traces: 16 items,
    0 traps. Base: 79 trap items, all 61 failures after a search at the call limit.
  - The data's search semantics: no lab service implements web_search and the eval always
    answers "unavailable", but in the training data it succeeds 60 times of 108 (web_research:
    6 questions x 10, each answered from the returned snippet, "Based on the PyTorch
    tensor documentation, ...") and fails only in web_fallback's 48, which then run
    --help. No record shows a failed search followed by an answer.
  - Data (MEASURED, `probes/g6w_build.py`, `results/research_dataset_g6w.json`,
    `results/g6w-build.json`): each web_research tool result becomes one of
    augment_research.WEB_FAILURES (seeded: unreachable 24, timeout 20, HTTP 503 16; the
    eval's string appears 0 times, as augment_research requires), and the answer keeps
    its content without the claimed source: "I couldn't search the documentation (the
    network is unreachable), so this is from memory and unverified: you pass
    `tensor.to(device, non_blocking=True)`. ... Check the PyTorch documentation for
    `Tensor.to` before relying on it." (lookup_failed's shape). 60 records changed
    (default 40, off 20, none traced), 1,188 identical; web_search successes 60 -> 0.
  - Dry run (MEASURED, `results/g6w-dry.log`): 1,347 sequences as G6u, 0 truncated (max
    2,047), 1,372,409 tokens (G6u 1,373,015), 224,663 trained (G6u 223,029); trace guard
    and `RENDER=think` parity PASS. The log differs from `results/g6u-dry.log` in those
    two counts only (plus attn_bf16's config line, which G6u's dry log lacks).
  - Time: 338 steps, about 56 min at G6u's 3,338 s (ARITHMETIC).
  - Run: `DATASET=/out/research_dataset_g6w.json RENDER=trace GUARD=hw MAX_LEN=2048
    probes/gpurun.sh g6w-train /probes/attn_bf16.py g6_train.py g6w-train`; eval and
    compare (gate, and `COMPARE_TO=g6u`) as G6u; then N1's repeats for G6w
    (`LABEL=g6w probes/noise_eval.sh results/g6w-train-adapter "r2 r3"`),
    `probes/noise_summary.py results/n1w-summary.json "r2 r3" base g6u g6w`,
    `probes/trap_failures.py` and `probes/trap_web_calls.py` over "r1 r2 r3" for g6u g6w.
  - **Pass rule, unchanged.** "Beyond the spread" is N1's: |mean difference| > the larger
    of the two ranges (noise_summary.py).
  - What it decides: G6w replaces G6u as the candidate if it passes both modes, has no row
    below base beyond the spread, is above G6u beyond the spread on at least one trap row
    (trap2, rocky_trap, trap3 noticed) and below it on none. Trap rows tie and web_search
    on trap items falls to 36 of 123 or fewer -> the search was not what held the traps;
    read the failures per item. web_search on trap items stays above 36 -> the habit
    comes with the traces' reasoning (base's), not the data's search results, and the
    data lever is spent. Any row below G6u beyond the spread and none above -> G6w is
    dropped, G6u stays.
  - Expectations (INFERENCE, written before the run so they can be wrong): web_search
    falls on every set, not only traps; truncated_in_think after a search falls, since
    60 records now close the think and answer after a failed search; trap rows rise 1-3
    items each, which may sit inside the spread. Risk: "I couldn't search ..., so this is
    from memory" opens trap answers without a denial in the first sentence (answered
    without denying rises), or appears after lookups that worked.
- **G6W RESULT (MEASURED, 2026-10-01; `results/g6w-{train,compare,compare-vs-g6u,
  web-calls,trap-failures,think-counts,tool-choice}.json`, `results/n1w-summary.json`,
  `results/research-eval-*-lightning-g6w-*.*`, `results/noise/*-g6w-*`): PASSES both
  modes at the gate, but by the rule G6w is dropped and G6u stays the candidate. Over
  three repeats G6w ties G6u on every trap row, fabricates on trap3 beyond the spread
  against base (3.00 vs 0, range 2) and trails G6u on promql beyond the spread (13.67 vs
  15.33, range 1). web_search on trap items falls only from 72 to 62 of 123 (> 36): the
  search habit comes with the traces' reasoning, so the data lever is spent.**
  - Train: 338/338 steps, 3,348 s, no guard abort, exit 0; loss 0.6098 first step, 0.0578
    last; epoch means 0.437 / 0.213 (G6u 0.436 / 0.214). Torch peak 19.94 GiB; max 79 C,
    153 W; sw_power_cap only. Trace guard 502/502 and parity 1658/1658 PASS in the run.
    Adapter `results/g6w-train-adapter/` (not in git).
  - Eval: 14 runs, exit 0 (base reused); 956 items, 0 turn errors, 937 calls all
    `xml_function`, 0 unparsed; 9 at the call limit, 1 think-trapped, 1 truncated (G6u
    12, 6, 2). N1 repeats r2 and r3: 10 runs, exit 0 (the script's own exit 1 is its last
    line, `[ "$with_base" = base ] && ...`, false without `base`).
  - Gate, thinking on vs base: held_out +12, seen_tool +7, trap +7, held_out2 +13,
    two_flag +5, task +6, trap2 +6, rocky_held_out +13, rocky_trap +6 (all wins);
    rocky_task 0.65 vs 0.65, promql +2, alert +3, trap3 noticed +3 and fabricated 4 vs 0
    (ties at the gate's line). vs G6u, single run: every row a tie thinking on; thinking
    off, trap3 noticed 0.583 vs 1.000 (-5, a loss). vs the Qwen3-8B v3 adapter, thinking
    off: 2 wins (promql, alert), 0 losses, 20 ties.
  - N1 means, thinking on (base / G6u / G6w): task 13.00 / 17.67 / 18.00; rocky_task
    12.67 / 12.67 / 11.67; trap2 3.00 / 9.00 / 9.00; rocky_trap 1.00 / 6.00 / 7.33;
    promql 12.00 / 15.33 / 13.67; alert 6.00 / 7.00 / 7.33; trap3 noticed 3.67 / 8.00 /
    7.67; trap3 fabricated 0 / 2.00 / 3.00; rocky_trap_control (wrong denials) 0 / 0.33 /
    2.00 (range 2, not beyond). G6w is steadier on task (18, 18, 18).
  - Mechanism (`probes/trap_web_calls.py`, r1-r3 pooled, G6u -> G6w): items with a
    web_search 167 -> 104 of 983; trap items 72 -> 62 of 123. Failures after a search:
    truncated_in_think 11 -> 1, call_limit 5 -> 9, fabricated 6 -> 5, answered without
    denying 5 -> 5. Across the trap rows (`probes/trap_failures.py`): truncated_in_think
    11 -> 1, call_limit 18 -> 21, trap3 fabricated 6 -> 9, answered without denying 5 -> 7.
    Reasoning on 544 of 971 thinking-on turns (G6u 555 of 994); unclosed think 6 -> 1.
  - Reading: the 60 failed searches did what they were built to do (the think closes and
    an answer follows a failed search) and not what was hoped (fewer searches). The
    closed think turned the think-trap failures into call-limit and fabrication failures,
    not passes. The pre-registered risk showed on trap3, read per item: when the lookup
    of a tool that does not exist fails, G6w writes "I couldn't check `gti --help` (the
    tool isn't installed here), so I can't verify the exact syntax" and then answers
    from memory; thinking on, 3 r1 items go from noticed (G6u) to fabricated, 2 of them
    after a failed web_search ("the online documentation couldn't be queried"), and a
    fourth (`ipa-healthcheckd --fix`) is fabricated by both. The new
    phrase itself is rare: "couldn't search / from memory / unverified" in 0 of 478
    thinking-on answers and 11 of 478 thinking-off (G6u 4; alert 7, rocky_task 3,
    rocky_trap 1); whether those followed a lookup that worked was not read.
  - promql, read per item over three repeats (G6u 46, G6w 41 of 54): four items differ,
    all on live metrics. One is the model's error (`up{job="gpulab"} == 0`, a job label
    that does not exist); the others (`up` count, laptop fan RPM, desktop VRAM total)
    read live values, and G6w ran two days after G6u, so drift in the live state may
    account for part of the gap (UNKNOWN, not separated).
  - Expectations against the result: web_search fell (overall, not per set, which was
    not counted); truncated_in_think after a search fell (11 -> 1); trap rows did not
    rise (0, +1.33, -0.33); the risk partly happened (fabrication up, answered without
    denying 5 -> 7 on the trap rows).

- **N2: thinking-off noise of the yardstick comparison, pre-registered 2026-10-01 before
  any repeat ran.** No training. The goal's second clause ("beats the Qwen3-8B v3
  adapter") rests on single thinking-off runs: G6u vs Qwen v3, 3 wins (rocky_task +6,
  alert +7, trap3 noticed +7), 0 losses, 19 ties. N1 measured thinking on only.
  - Runs: G6u thinking off, all seven sets, r2 and r3, on the gate's server (`THINK=off
    LABEL=g6u probes/noise_eval.sh results/g6u-train-adapter "r2 r3"`); Qwen v3 thinking
    off, all seven sets, r2 and r3 (`probes/n2_qwen.sh "r2 r3"`), on the server the
    yardstick ran on (session 09256598's docker command; adapters as r1's provenance
    lists them; the v3 file's sha256 matches r1's, 5b574180513c). Settings as the gate's
    thinking-off runs: 512 tokens, 3 calls, temperature 0, concurrency 16, window 4000,
    seed 20260923. r1 is the existing runs. Laptop only; the pool is not touched.
  - Pre-check (MEASURED): Qwen's r1 was scored by harness d5ce5b2f (uncommitted,
    2026-09-23), every Lightning run by a56bfe79. Rescoring copies of the seven r1 files
    with the current scorer changes 0 headline rows. The run loop may still differ:
    r1 against r2-r3 shows it.
  - Summary: `MODE=nothink probes/noise_summary.py results/n2-summary.json "r2 r3" g6u
    qwenv3` (the default, thinking-on path reproduces n1-summary.json and
    n1w-summary.json exactly).
  - What it decides: G6u beats Qwen v3 thinking off at the noise level if it is above
    Qwen beyond the spread on at least one row and below it on none. Any row below
    beyond the spread -> the clause does not hold as stated for thinking off; name the
    row. If Qwen's r1 sits outside its r2-r3 range on several rows, the rebuilt server or
    harness differs from r1's; then the verdict uses r2-r3 only and says so.
  - Expectations (INFERENCE): Qwen v3 is steady (memory: ~1 of 158 items between
    repeats); G6u's thinking-off ranges are 0-2 items; the three wins (6-7 items each)
    survive.
- **N2 RESULT (MEASURED, 2026-10-02; `results/n2-summary.json`, `results/n2-{g6u,qwen}.log`,
  `results/noise/*-nothink-r{2,3}.*`; 28 runs, all exit 0): as pre-registered, the clause
  does not hold. G6u is above Qwen v3 beyond the spread on six rows and below it on one,
  held_out: 88.33 vs 89.67 of 90 (range 1), 1.5 points. The rule said "below on none".**
  - Above Qwen v3 beyond the spread (G6u / Qwen means, larger range): trap3 noticed 12.00 /
    5.00 (0); alert 6.00 / 0 (2); rocky_task 9.00 / 4.67 (4); promql 13.33 / 9.33 (2); trap
    15.00 / 13.00 (0); trap2 11.00 / 9.67 (1). The other 15 rows: within the spread.
  - The rebuilt Qwen server reproduces r1: Qwen's r1 sits inside its r2-r3 range on every
    row (e.g. held_out 90 / 89 / 90, promql 9 / 10 / 9, rocky_task 5 / 5 / 4), so the
    verdict uses r1-r3 and the harness difference (d5ce5b2f vs a56bfe79) shows no effect.
  - Spread, thinking off: Qwen v3 0-1 items on every row (as expected); G6u 0-2 except
    rocky_task 4 (11, 7, 9: the gate's 0.55 was a high draw; mean 0.45). trap3 noticed is
    12/12 in all three G6u repeats.
  - held_out per item (r1-r3): G6u misses `held_out-patch-5` in all three (it answers
    `--backup-if-mismatch`, grounded in `patch --help`, the wrong flag for the
    question) and `held_out-file-5` in two; Qwen v3 misses only `held_out-file-3`, once,
    after its lookup failed. The gap is one item G6u gets wrong every time.
  - Reading: on what it was trained for (flag lookup) G6u sits a point below Qwen v3,
    which is at its ceiling (89.67 of 90); on the task, tool and trap rows it is
    clearly ahead. Whether "below on none" was the right bar for a 90-item row at a
    ceiling is David's call; the rule as written is not met.

### G7: serving on vLLM 0.29 (laptop)
- Base NVFP4, then base + LoRA, then merge if needed. UNKNOWN:
  - whether vLLM's nemotron_h supports LoRA on placement A's modules (read the vLLM
    source before G6);
  - how much the train/serve precision mismatch (NF4 while training, NVFP4 while serving)
    costs. G6 measures that cost, because its eval runs through the server.
- Merge path (fallback): merge the LoRA into BF16, then re-quantize. Which tool does that
  for nemotron_h is UNKNOWN.
- **LoRA smoke (MEASURED, 2026-09-27; `probes/g7_lora_adapters.py`, `g7_lora_smoke.sh`,
  `g7_lora_smoke_client.py`; `results/g7-lora-smoke/`): vLLM 0.29 applies LoRA to every
  placement-A family.** Untrained seeded adapters (r=16, alpha 32; PEFT's own key names,
  `base_model.model.model.layers.N.mixer...`; 93 modules = 24 attention + 23 `in_proj` +
  46 shared-expert) served beside base NVFP4 with G7a's flags plus `--enable-lora
  --max-lora-rank 16 --max-loras 1`:
  - base twice: bit-identical (noise 0); `zero` (lora_B = 0): identical to base, so it loads;
  - `attn`, `inproj`, `shexp`, `full`: each changes the greedy tokens (max log-prob shift
    0.25, 0.64, 0.24, 0.29), so none is skipped silently;
  - no LoRA warnings in the log; NVFP4 weights loaded in 4.89 s from the laptop's mirror
    (G7a: 76 s over NFS, cold); card 22.9 of 23.9 GiB used.
  This settles the first UNKNOWN above. Still open: how closely a trained adapter served on
  NVFP4 tracks the NF4 model it was trained on (G6 measures that through the server).

### G7b: the candidate served on the desktop 3090 (sm_86), pre-registered 2026-10-02
- Why: every Lightning serve so far ran on the laptop (sm_120), which is also the only
  card that trains. Replacing Qwen3-8B as the lab's default adapter base needs the
  adapter servable where the lab serves, and the front door (LiteLLM) is on the desktop.
  Whether vLLM 0.29 runs NVFP4 with `--moe-backend marlin`, fp8 KV and LoRA on sm_86 is
  UNKNOWN (nothing in this plan or memory measures it).
- Run: `LABEL=g6u probes/g7b_desktop.sh results/g6u-train-adapter`: the gate's server
  flags and image (`vllm/vllm-openai:v0.29.0`, already on the desktop; the NVFP4
  snapshot bee7596 is in the desktop's cache, 21 GB; no download), bound to lab-desktop
  only; the seven thinking-off sets from the laptop through `probes/g7a_eval.py`, as the
  gate. After the 14B pool sweep (gpu-lab branch pool-14b-prefill) frees the card.
- If the server refuses one flag as unsupported on sm_86, that flag is dropped once
  (`DROP=...`) and the result says so; an OOM is recorded, not retried (hard stops).
- What it decides: (1) serves or not (loads, LoRA applies, all calls `xml_function`, 0
  unparsed); (2) per row, the desktop run against G6u's three thinking-off laptop runs
  (N2): rows outside the laptop's [min, max] are named with their distance. A row more
  than 2 items outside on a set means the sm_86 path changes the model's behaviour and
  serving there is not equivalent.
- Expectations (INFERENCE): it serves (Marlin's FP4 path is a weight-only dequant that
  is not Blackwell-specific); fp8 KV is the likeliest refusal; numbers inside the
  laptop's range on most rows, slower per set.
- **Attempt 1 (MEASURED, 2026-10-02 00:27; `results/g7b/g6u-desk-serve-attempt1.log`,
  `results/g7b/g6u-attempt1.log`): does not serve as is.** Every flag was accepted
  (fp8 KV included) and the NVFP4 parts load through Marlin's weight-only path ("Your
  GPU does not have native support for FP4 ... Marlin"). The checkpoint is mixed:
  vLLM detects ModelOpt FP8, NVFP4, W4A16_NVFP4 and MXFP8 sections. For the FP8 linears
  it selects `CutlassFP8ScaledMMLinearKernel`, and the first profiling forward, in a
  Mamba mixer's `in_proj` (under the LoRA wrapper), dies: `RuntimeError:
  cutlass_scaled_mm_sm80_epilogue ... scaled_mm_c2x.cu:89`. Cause, read in the image:
  that kernel's `is_supported` (`kernels/linear/scaled_mm/cutlass.py:164`) returns True
  on any CUDA device with no compute-capability gate, and it is listed before
  `MarlinFP8ScaledMMLinearKernel` in `_POSSIBLE_FP8_KERNELS`; sm_86 has no FP8 tensor
  cores. Not a flag refusal, so the DROP fallback does not apply. (An upstream
  candidate of the kind `~/gpu-lab/docs/contributions.md` tracks.)
- **Amendment, written before attempt 2:** one retry with `--linear-backend marlin`
  (`EXTRA=--linear-backend marlin`), vLLM 0.29's per-layer kernel selector and the
  linear-layer twin of the gate's `--moe-backend marlin`; for layer types Marlin has no
  kernel for it falls back to automatic selection with a warning. It also changes the
  FP8 linears from W8A8 to weight-only, so any row outside the laptop's range carries
  that difference too. If attempt 2 fails, G7b's result is "not servable on sm_86
  with vLLM 0.29 without a source change", and it stops there.
- **G7B RESULT (MEASURED, 2026-10-02; `results/g7b-compare.json`, `results/g7b-g6u.log`,
  `results/g7b/`): attempt 2 SERVES, and the desktop's answers match the laptop's: 0 of
  22 rows more than 2 items outside the laptop's three thinking-off runs.** The
  candidate can be served from the desktop's 3090 with the gate's flags plus
  `--linear-backend marlin`.
  - Server: ready in 112 s; `MarlinFP8ScaledMMLinearKernel` for the FP8 linears,
    `MarlinNvFp4LinearKernel` for NVFP4 GEMMs, the Marlin NVFP4 MoE backend, FlashInfer
    attention; fp8 KV accepted. Card 21.2 GiB at ready, 21.4 GiB, 60 C, 127 W after.
  - Eval: 7 sets, exit 0; 478 items, 476 answered, 1 at the call limit, 1 truncated; 467
    calls all `xml_function`; 0 final answers holding an unparsed `<tool_call>`.
  - Rows outside the laptop's [min, max] (desktop vs laptop r1-r3): two_flag 7 vs 9-10
    (-2); rocky_held_out 54 vs 55-56 (-1); no_tool over-trigger 2 vs 0-1 (one more);
    task 15 vs 12-14 (+1); promql 15 vs 12-14 (+1); alert 8 vs 5-7 (+1). Mixed in sign,
    none past the pre-registered 2-item line. The other 16 rows sit inside the range.
  - Speed: slower per set on the 3090: v2 190 s vs 90-93 s, alert 28 vs 15-21, v1 142 vs
    113-132; rocky about equal (88 vs 81-86).
  - Reading: the NVFP4 checkpoint serves on sm_86 only through Marlin for every
    quantized layer type; vLLM 0.29's automatic choice for the FP8 linears is wrong
    there (attempt 1). With that one flag the adapter behaves as on the laptop at the
    eval's resolution, so the desktop can serve the candidate while the laptop trains.
    The adapter copy it served from is `llm:/home/david/g7b/g6u/` (180 MB).

### G7c: base Lightning pooled across both cards, pre-registered 2026-10-02
- Why: the lab exists to pool VRAM; the pool has served only Qwen3-14B and Llama-70B.
  G7b showed Lightning loads on sm_86 with `--linear-backend marlin`; the pool image
  (`gpu-lab:vllm-ray`) is vLLM 0.29.0 and `nemotron_h` implements `SupportsPP`.
  Whether Lightning runs pipeline-parallel across sm_86 + sm_120 is UNKNOWN.
- Run: `probes/g7c_pool.sh`: `lab pool up` with `POOL_MODEL` = the NVFP4 repo,
  `POOL_MAXLEN=16384`, and `POOL_EXTRA_FLAGS` = the gate's Lightning flags plus
  `--linear-backend marlin` (bin/lab's own pool flags otherwise: PP 2 over Ray,
  `--no-enable-flashinfer-autotune`, gpu util 0.92, 512 batched tokens). Base model only:
  the pool containers mount only `/srv/model-cache`, and putting the adapter there means
  writing the shared mirror, which waits for David. Then the head's KV lines, base v1
  thinking off (against base's laptop run on G6's server), and bench.py at c=1 and c=16.
  Always `lab pool down`, then `lab up` on the desktop.
- What it decides: (1) Lightning pools or not; if not, the error, recorded, no retry
  beyond this design; (2) the pooled KV budget against the single laptop card's; (3) v1
  rows within 2 items of the laptop run (one run each: descriptive, not a verdict);
  (4) decode tok/s at c=1 and c=16 as the first pooled-Lightning baseline.
- Expectations (INFERENCE): it serves; KV well above one card's (most of each card is
  free once the weights split); single-stream decode slower than one card (one link
  crossing per token, ~0.24 ms, plus the pipeline bubble), c=16 aggregate higher.
- **G7C RESULT (MEASURED, 2026-10-02 00:46-01:01; `results/g7c.log`, `results/g7c/`):
  OOM on the laptop's stage; not retried (hard stops).** Ray placed both stages and both
  loaded through Marlin (FP4, FP8 and MoE): weights 9.01 GiB on the desktop (PP0) and
  9.71 GiB on the laptop (PP1); profiling gave 12.34 and 11.28 GiB of KV. Then the
  laptop's worker died in a forward pass, in the Mamba chunk scan's state passing
  (`ssd_state_passing.py` `_state_passing_fwd`, from `mamba_chunk_scan_combined_varlen`):
  "Tried to allocate 512.00 MiB ... this process has 22.74 GiB memory in use ... 402.75
  MiB is free". The endpoint never answered (900 s), `lab pool down` exit 0, desktop
  `lab up` exit 0, llama-swap active again.
  - Reading: the pool ran bin/lab's settings, not the gate's: gpu util 0.92 (bin/lab's
    note: "about as high as the laptop can go") and vLLM's default max_num_seqs (256),
    where the single-card server runs 0.85 and `--max-num-seqs 16`. The SSD state buffer
    grows with the number of sequences, and profiling did not leave room for it.
  - Proposed next (David's call, since it moves thresholds after an OOM): the same run
    with `--max-num-seqs 16` in `POOL_EXTRA_FLAGS` and `POOL_GPU_UTIL=0.85`.

## Hard stops and rules

- No edits to `~/gpu-lab` until the feasibility verdict. Probes and results stay in this
  repo. (Verdict given 2026-09-25: feasible; see the top.)
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

### F1 rerun (determinism check) — CHOSEN before the run, 2026-09-25
- Config: F1 unchanged (g5_followups.py, no env), label g5-attnbf16-lean-cce-rerun-laptop.
- CHOSEN reading: every step's loss equal to F1's to all printed digits at 512/1024/2048
  -> runs are deterministic, so F2's later-loss drift comes from its backward (grad
  summation order). Any step differing -> run-to-run nondeterminism; F1-vs-variant loss
  differences of that size carry no signal.
- RESULT (MEASURED; corrected 2026-09-25 — the first write-up said "68 of 72" and missed
  the spike below): 68 of 69 step losses differ from F1 (23 per length; only step 0 @512
  is equal; step 1 @512: F1 1.988, rerun 1.980, F2 1.984). -> Run-to-run
  NONDETERMINISM, by the CHOSEN rule.
  - @512 the rerun is plain noise: max |rerun − F1| 0.026, mean +0.002. F2's drift is
    the same size (max 0.021, mean +0.001), so the lean backward's drift is noise too.
  - @1024 the rerun has a LOSS SPIKE no other run shows (F1, F2, all three F3 variants
    open 1024 at 1.06-1.37): step 0 3.358 (F1 1.369), peak 3.851 at step 1; from step 10
    on it runs 0.005-0.13 above F1. The last 512 step's loss was normal (1.684 vs F1
    1.683) and the next loss was 3.358, so the damage was done by that one update (the
    model and optimizer carry across lengths).
  - @2048 the rerun stays above F1 on all 23 steps (mean +0.045, max +0.100). Inferred
    to be the spike's after-effect, not base noise: @512 the noise has mixed signs (10 of
    23 steps above F1). Its last-5 mean @2048 is 0.698 vs 0.667.
  - Cause UNKNOWN (grad norms are not logged). g5_train_step.py has no gradient clipping,
    no LR warmup, constant lr 1e-4, batch 1, 8-bit paged Adam, so one outlier gradient
    lands unclipped. qlora.py does not share this: Trainer clips at max_grad_norm 1.0
    (SOURCED: transformers 5.16.1 default in gpu-lab:training), warms up and logs
    grad_norm each step. G6 should keep clipping and read that grad_norm log.
  - Reading of the F3 training-loss table: the base noise (≤0.026 per step @512) is well
    under the F3 gaps (last-5 means 0.06-0.17 below F1's; 0.05 between neighbouring
    variants), but one of the six runs on this
    data spiked and that moved its last-5 mean by 0.031, so the table stays a loss-stream
    curiosity, not an eval.

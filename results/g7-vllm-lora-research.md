# G7 research: LoRA on Lightning NVFP4 under vLLM 0.29.0 (sm_120, --moe-backend marlin)

Written 2026-09-26. Read-only; no GPU used; no server started; nothing in ~/moe-qlora or
~/gpu-lab edited.

Source: `vllm/vllm-openai:v0.29.0` (image c29147676055), Python tree extracted with
`docker run --rm --entrypoint bash ... tar --exclude=*.so` into `g7/src/vllm/` next to this
file. Paths below are relative to `/usr/local/lib/python3.12/dist-packages/vllm/` in the
image. Tags: SOURCED (file:line), MEASURED (I ran it), ARITHMETIC, UNKNOWN.

## 0. What is being served (MEASURED from the checkpoint on disk)

`/srv/model-cache/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/bee75962.../hf_quant_config.json`
is `quant_algo: MIXED_PRECISION`, not plain NVFP4. Per module:

| module (HF name) | count | serve precision | train precision (placement A config) |
|---|---|---|---|
| attention q/k/v/o_proj | 6 layers | **bf16 (in exclude list)** | bf16 (attn_bf16 verdict config) -> match |
| mamba in_proj, out_proj | 23 + 23 | **FP8** (W8A8 static) | in_proj NF4 |
| shared_experts up/down_proj | 23 + 23 | W4A16_NVFP4, group 16 | NF4 |
| routed experts.{e}.up/down_proj | 2944 + 2944 | W4A16_NVFP4, group 16 | NF4 (Route 1) |
| lm_head | 1 | **W4A16_NVFP4** | bf16 |
| mixer.gate (router), conv1d, embeddings | - | bf16 (excluded) | bf16 |

vLLM resolves this as `modelopt_mixed` (G7a log line 23, MEASURED) ->
`ModelOptMixedPrecisionConfig.get_quant_method` (model_executor/layers/quantization/modelopt.py:2408-2460):
FP8 -> `ModelOptFp8LinearMethod`, W4A16_NVFP4 linear -> `ModelOptNvFp4W4A16LinearMethod`,
W4A16_NVFP4 experts -> `ModelOptNvFp4FusedMoE` with `use_a16=True` (modelopt.py:1425) SOURCED.

Config (MEASURED): 52 layers = 23 mamba + 23 moe + 6 attention; 128 experts, top-6, H=2688,
I=1856, shared I=3712, relu2 (non-gated), n_groups 8, no `moe_latent_size`, untied lm_head,
vocab 131072.

G7a serve flags (MEASURED, `results/g7a-serve-final.log` line 8): `--max-model-len 16384
--enforce-eager --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 --mamba-cache-mode align
--max-num-seqs 16 --moe-backend marlin`, rev bee75962. Weights 17.86 GiB, KV 1.5 GiB (line 105),
"consumed 18.23 GiB, peak activation 0.16 GiB" (line 118).

## Q1. Does nemotron_h declare LoRA support, and for which modules?

- **Yes.** `NemotronHForCausalLM(..., SupportsLoRA, ...)` (models/nemotron_h.py:694-706) SOURCED.
  - `packed_modules_mapping = {"qkv_proj": ["q_proj","k_proj","v_proj"]}` (nemotron_h.py:720-726).
  - `embedding_modules = {"embed_tokens": "input_embeddings", "lm_head": "output_embeddings"}` (:729-732).
  - `lora_skip_prefixes = ["mtp."]` (:735); `is_non_gated_moe = True` (:708);
    `is_3d_moe_weight` not set -> default False (models/interfaces.py:699).
  - LoRA name mapping uses `hf_to_vllm_mapper.get_rename_mapper()` (lora/worker_manager.py:137;
    models/utils.py:163-182): `backbone.` -> `model.`, stacked q/k/v renames dropped so q/k/v stay
    separate adapter entries. SOURCED.
- vLLM has no per-model LoRA allow-list: every `LinearBase` or `MoERunner` suffix is a supported
  LoRA module (lora/utils.py:220-237 `get_supported_lora_modules`; :240-272). So, per module
  (SOURCED unless tagged):

| module | vLLM class | LoRA wrapper | status |
|---|---|---|---|
| q/k/v_proj | `qkv_proj` QKVParallelLinear (nemotron_h.py:437-444) | MergedQKVParallelLinearWithLoRA (packed list len 3) | supported |
| o_proj | RowParallelLinear (:445-451) | RowParallelLinearWithLoRA | supported |
| mamba in_proj | MergedColumnParallelLinear, 5 output slices [4096,4096,1024,1024,64] since n_groups 8 % tp 1 == 0 (layers/mamba/mamba_mixer2.py:332-344) | MergedColumnParallelLinearVariableSliceWithLoRA (lora/layers/column_parallel_linear.py:693-765): packed list empty + >=3 slices -> takes one PEFT lora_A, duplicates it per slice, splits lora_B by output_sizes | supported. HF split order is gate, xBC, dt (transformers 5.16.1 modeling_nemotron_h.py:394, 469, 508) = vLLM order; 10,304 rows both (ARITHMETIC) |
| mamba out_proj | RowParallelLinear (mamba_mixer2.py:470-477) | RowParallelLinearWithLoRA | supported by vLLM (PEFT refuses it in training, plan) |
| mamba conv1d | MergedColumnParallelLinear, 3 slices (mamba_mixer2.py:320-330) | would be wrapped by the VariableSlice class if not excluded; the mixer never calls conv1d.forward (uses `conv_weights`, :445-449), so a conv1d LoRA would be silently inert (inference from source) | exclude |
| shared_experts up/down_proj | ColumnParallelLinear / RowParallelLinear (nemotron_h.py:100-116, 175-184) | ColumnParallelLinearWithLoRA / RowParallelLinearWithLoRA | supported |
| routed experts | `experts` MoERunner via FusedMoEFactory (nemotron_h.py:206-228) | FusedMoEWithLoRA (2D) — see Q2 | supported with Marlin |
| router `mixer.gate` | GateLinear(ReplicatedLinear) | **skipped** for non-gated MoE (lora/model_manager.py:448-455) | not supported |
| lm_head | ParallelLMHead | LogitsProcessorWithLoRA (model_manager.py:498-519; lora/layers/logits_processor.py); base logits via the quantized head's own `_apply_head` (logits_processor.py:175); vocab limit 258048 (:91) | supported |
| embed_tokens | VocabParallelEmbedding | VocabParallelEmbeddingWithLoRA | supported |

- Default (`--lora-target-modules` unset) wraps **all** of the above, including every MoERunner
  (model_manager.py:421-432, 713-737). `--lora-target-modules` (LoRAConfig.target_modules,
  config/lora.py:49; engine/arg_utils.py:1499-1500) restricts by suffix; packed children match
  their parent (lora/utils.py:275-315). With the flag set, a matched module that cannot be
  wrapped raises instead of warning (model_manager.py:531-535).
- Adapter key check: non-expert keys must end in a supported suffix; expert keys must match
  `experts.{e}.up_proj|down_proj` exactly (lora/lora_model.py:218-240; worker_manager.py:108-118).
- PEFT adapter-level fields only: `r`, `lora_alpha`, `target_modules`, `use_rslora`, `use_dora`
  (DoRA rejected), `modules_to_save` must be None, bias "none" (lora/peft_helper.py:22-37, 45-51,
  79, 120-132). `rank_pattern`/`alpha_pattern` are dropped by `from_dict` (:79), so one
  scaling `alpha/r` (:60) applies to every module. With alpha = 2r everywhere (plan F3),
  r=8 experts + r=16 elsewhere still scale by 2 — consistent (ARITHMETIC).

## Q2. Routed-expert LoRA: supported? with Marlin? with NVFP4? adapter layout?

- **Yes, in principle, and Marlin is the one NVFP4 backend that can do it.**
  - `FusedMoEWithLoRA.__init__` requires a modular (non-monolithic) kernel whose experts class
    `supports_lora()` (lora/layers/fused_moe.py:39-41, 64-87). SOURCED.
  - `MarlinExperts(LoRAExpertsMixin, MarlinExpertsBase)` (fused_moe/experts/marlin_moe.py:694);
    the mixin flips `supports_lora()` to True (experts/lora_experts_mixin.py:31-33). Its
    `apply()` has an explicit LoRA path: w13 LoRA injected in the activation callback, w2 LoRA
    in `moe_sum` (marlin_moe.py:757-910). LoRA math is bf16 punica, independent of weight format.
  - NVFP4 oracle: backend MARLIN -> `[MarlinExperts]` (fused_moe/oracle/nvfp4.py:127-132);
    `MarlinExpertsBase` accepts `use_nvfp4_w4a16` and `kNvfp4Static` (marlin_moe.py:583-588, 614-634).
    G7a log confirms MARLIN was selected (log line 33, MEASURED).
  - Every other NVFP4 MoE backend lacks LoRA: emulation returns "kernel does not support LoRA"
    (experts/nvfp4_emulation_moe.py:381-393); B12x and flashinfer_cutlass raise
    (fused_moe/b12x.py:770; experts/flashinfer_cutlass_moe.py:396). Searched: `grep -rn
    LoRAExpertsMixin` -> only marlin_moe, triton_moe, trtllm_lora_moe (bf16), gpt_oss_triton. SOURCED.
  - Non-gated path exists and nemotron_h is its only user: `_w13_slices = 1` when not
    act-and-mul (fused_moe.py:53-55); pairs padded to triplets (model_manager.py:552-565,
    809-817); w3 reuses w1 with scaling 1 (lora/lora_weights.py:185-215). `grep is_non_gated_moe
    = True` -> only nemotron_h.py:708. The model_manager comment says a working NemotronH
    adapter came from PEFT `target_modules="all-linear"` (model_manager.py:444-447). SOURCED.
  - Marlin pads I only if needed: K=2688 is a multiple of 128, so padded_N = round_up(1856, 64)
    = 1856 (marlin_utils_fp4.py:374-378; ARITHMETIC) -> LoRA-B rows and Marlin N agree.
  - The CUDA helper `moe_lora_align_block_size` (punica_gpu.py:404) is compiled for sm_120:
    `cuobjdump -xelf all _moe_C_stable_libtorch.abi3.so` -> `...7.sm_120.cubin` contains
    `moe_lora_align_block_size_kernel` (MEASURED). The shrink/expand kernels are Triton
    (lora/ops/triton_ops/fused_moe_lora_op.py); Triton 3.7.1 runs on this card (G7a's Triton SSU,
    MEASURED in plan). Whether `fused_moe_lora` JIT-compiles and is correct on sm_120: UNKNOWN
    until R4.
- **Expected adapter layout = 2D, one PEFT Linear per expert** (SOURCED):
  - `packed_modules_mapping["experts"]` = names from `NemotronHModel.get_expert_mapping`
    (nemotron_h.py:672-689: gate="up_proj", down="down_proj", up="") filtered of the empty w3
    (lora/utils.py:401-410): `experts.0.up_proj, experts.0.down_proj, experts.1.up_proj, ...`
    (256 entries; routed_experts.py:1129-1143 for the format).
  - So the safetensors keys must be
    `base_model.model.model.layers.{L}.mixer.experts.{e}.up_proj.lora_A.weight` (r, 2688),
    `...up_proj.lora_B.weight` (1856, r), `...down_proj.lora_A.weight` (r, 1856),
    `...down_proj.lora_B.weight` (2688, r). `backbone.` instead of `model.` also maps.
  - vLLM stacks them to (E, r, in)/(E, out, r) (lora_weights.py:155-229) into buffers
    (max_loras, E, r, H) etc. (fused_moe.py:162-222).
- **Mapping from probes/expert_lora.py** (ARITHMETIC; its convention is
  `up += s·(x@A_up[e])@B_up[e]`, PEFT's is `y += s·x@Aᵀ@Bᵀ`):
  `up_proj.lora_A = A_up[e].T`, `up_proj.lora_B = B_up[e].T`,
  `down_proj.lora_A = A_down[e].T`, `down_proj.lora_B = B_down[e].T`. Shared mode: replicate the
  single factor pair to all 128 experts (exact). expert_lora.py itself never saves a PEFT file
  ("PEFT will not save these; use expert_lora_state_dict()", its docstring), so a converter is needed.
- **What PEFT `target_parameters` would produce is the wrong layout** (SOURCED + ARITHMETIC):
  a 3D file (`experts.base_layer.lora_*` = first param, `experts.lora_*` = second, (r·E, in)/
  (out, r·E)). vLLM only reads that via `--enable-mixed-moe-lora-format` + `is_3d_lora_weight`
  (model_manager.py:836-845), and `_convert_3d_to_2d_moe_lora` then assumes a *gated*
  gate_up and splits LoRA-B in half (model_manager.py:1036-1048). For non-gated Lightning that
  keeps rows 0..927 of up_proj's B as w1 and drops 928..1855 into the unused w3: silently wrong.
  Moot anyway: target_parameters crashes in training (plan, hand-off #4). Use the 2D names.
- Optional `--enable-moe-shared-loras` (config/lora.py:81-86): "shared-outer" layout, only w13
  lora_A and w2 lora_B shared; keys `experts.w1/w2/w3` pre-stacked (lora/utils.py:391-400;
  lora_weights.py:231-260 asserts all three present, so the non-gated model still needs a w3
  entry). Saves GPU buffer memory (below) but is fragile here; not for the first rung.
- **GPU memory for expert LoRA buffers** (ARITHMETIC, max_loras 1, max rank 16, bf16):
  128·16·(2688+1856+1856+2688)·2 B = 35.5 MiB per MoE layer × 23 = **0.80 GiB**, allocated
  whenever `experts` is a LoRA target, adapter or not. Shared-outer flag: 0.33 GiB. Placement A
  buffers: 29 MiB; lm_head 4 MiB. Against G7a's budget (0.85 × 23.4 = 19.89 GiB; 18.23 consumed
  + 0.16 activation; KV 1.5 GiB) KV falls to about 0.7 GiB with experts, still above the
  0.14 GiB one 16384-token request needs (G7a attempt 1). ARITHMETIC; the LoRA activation peak
  is not counted.
- On-disk adapter size (ARITHMETIC): per-expert r=16 bf16 0.80 GiB, r=8 0.40 GiB; shared r=16
  replicated 0.80 GiB. vLLM casts to `lora_dtype` (auto = bf16) at load (lora_model.py:155-159).

## Q3. LoRA on Mamba in_proj/out_proj with a quantized base; restrictions

- In this checkpoint in_proj/out_proj are **FP8, not NVFP4** (section 0, MEASURED). Every
  linear LoRA wrapper runs `base_layer.quant_method.apply(...)` first and adds the punica LoRA
  term to its output (lora/layers/base_linear.py:204-208, 210-236), so the base format does not
  matter to the wrapper. SOURCED. Input to in_proj is the RMSNorm output and to out_proj the
  gated-norm output, both fresh tensors (mamba_mixer2.py:555, 582-586); the Mamba1 contiguity
  caveat (mamba_mixer.py:205-208) does not apply to Mamba2. Correctness on this card: UNKNOWN until R2/R3.
- Restrictions (SOURCED):
  - `--max-lora-rank` ∈ {1, 8, 16, 32, 64, 128, 256, 320, 512} (config/lora.py:27, 35); adapter
    r > max rank is rejected (peft_helper.py:124).
  - `--max-loras` default 1; `--lora-dtype` auto = model dtype (config/lora.py:136-137).
  - Flags exist: `--enable-lora --max-loras --max-lora-rank --lora-dtype --max-cpu-loras
    --fully-sharded-loras --lora-target-modules --specialize-active-lora
    --enable-mixed-moe-lora-format --enable-moe-shared-loras` (engine/arg_utils.py:1480-1513);
    `--lora-modules name=path` (entrypoints/launchers/cli_args.py:75, 212-213).
  - CUDA graphs: `cudagraph_specialize_lora=True` default captures separate graphs per active-LoRA
    count (config/compilation.py:660); irrelevant under `--enforce-eager`, which G7a needs for
    memory anyway (plan G7a item 1).
  - LoRA is incompatible with adaptive speculative verification (config/vllm.py:2651-2655);
    G7 runs no speculation.
  - Searched for Mamba/hybrid/prefix-cache/fp8-KV LoRA restrictions: `grep -rni lora` over
    config/cache.py, config/model.py, config/scheduler.py, v1/core/sched/scheduler.py,
    layers/mamba/, and `grep -rni 'not supported.*lora|lora.*not supported|not compatible'`
    over the tree -> none apply to this model (only Mamba1 contiguity, gate skip, B12x,
    flashinfer_cutlass, monolithic kernels, lm_head gather). Absence of a guard is not proof it
    works: UNKNOWN until run.
- punica on sm_120: platform returns PunicaWrapperGPU (platforms/cuda.py:561-562); linear LoRA
  ops are Triton; the one CUDA op for MoE LoRA is built for sm_120 (MEASURED above).

## Q4. Adapters on disk for a smoke test

- **None for Lightning** (MEASURED). `find` over /srv/model-cache, /home/david and
  /tmp/claude-1000 for `adapter_config.json`, `*.safetensors`, `*.pt` outside the HF hub finds
  only Qwen3 adapters (/srv/model-cache/adapters/qwen3-{8b,14b,32b}-*; e.g.
  qwen3-8b-research-v3: PEFT 0.21.0, r 16, alpha 32, 174.7 MB, wrong base). The G5/F1-F3
  probes never save weights: `grep save|state_dict|safetensors` over g5_train_step.py,
  g5_followups.py, f3_check.py, h6_*.py -> nothing; only g4_smoke_tiny_checkpoint.py saves, a
  tiny model to a container path.
- So G7 must first **make** adapters: a synthetic one for the machinery (CPU), and G6 (or a
  short run of g5_followups.py plus a save) for the real one. The training model's prefix is
  `model.` (transformers 5.16.1 `base_model_prefix = "model"`, `self.model = NemotronHModel`,
  modeling_nemotron_h.py:947, 1114; MEASURED in gpu-lab:training), so PEFT keys will be
  `base_model.model.model.layers.{L}.mixer.<name>.lora_{A,B}.weight`, which vLLM parses
  (lora/utils.py:156-208).

## Q5. Proposed G7 ladder (smallest first)

Base command for every rung = G7a's (MEASURED line 8) inside `vllm/vllm-openai:v0.29.0`:
```
--model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --revision bee7596271d1495f6992ae224aefde4410e816b8
--served-model-name lightning-nvfp4 --max-model-len 16384 --max-num-seqs 16
--gpu-memory-utilization 0.85 --kv-cache-dtype fp8 --mamba-cache-mode align
--moe-backend marlin --enforce-eager
```
LORA_A = `--enable-lora --max-loras 1 --max-lora-rank 16 --lora-target-modules qkv_proj o_proj in_proj up_proj down_proj`
(excludes experts, out_proj, conv1d, lm_head, embed_tokens: nothing unused gets buffers).

Thresholds below are proposals; David makes them CHOSEN before each run.

- **R0 (CPU only, no server): make the adapters.** Build the HF model on `meta` in
  gpu-lab:training, apply the placement-A PEFT config, read the real key names and shapes, and
  write safetensors with real CPU tensors plus an adapter_config.json (r 16, alpha 32,
  target_modules as trained):
  - `g7-null-A`: lora_A random, lora_B = 0.
  - `g7-rand-{qkv,o,in,shared}`: B random on one module kind only.
  - `g7-null-AE` / `g7-rand-E`: add `experts.{e}.{up,down}_proj` 2D keys (mapping in Q2).
  PASS: files written; key count = 93 modules × 2 (placement A, plan G5 debug run) plus
  23·128·2·2 = 11,776 expert tensors.
- **R1: LoRA machinery on, no adapter.** Base + LORA_A.
  PASS: server ready; log shows "Using PunicaWrapperGPU"; "Available KV cache memory" ≥ 0.14 GiB;
  G7a's smoke (2 items per split) with 0 `error` items. FAIL: any exception while wrapping
  (e.g. model_manager.py:531) or OOM. Record KV, load time.
- **R2: null placement-A adapter.** R1 + `--lora-modules g7null=<path>`.
  PASS: loads with no "expected target modules ... but received" (lora_model.py:234-240);
  `/v1/models` lists it; for 3 fixed prompts sent one at a time (max_tokens 1, temperature 0,
  prompt_logprobs), g7null and base give **bitwise equal** prompt logprobs (B = 0 adds exact
  zeros; one request at a time avoids the batch nondeterminism the SSU runs saw).
- **R3: is each module kind live?** R2 with the four `g7-rand-*` adapters.
  PASS: each changes the prompt logprobs (max |Δ| > 1e-3) and R2's base stays unchanged.
  Optional stronger check for bf16 attention only: merge ΔW into a copy of the attention shards
  (other shards symlinked) and compare to the served LoRA (e.g. max |Δlogprob| ≤ 0.05).
- **R4: routed-expert LoRA.** LORA_A with `experts` added to `--lora-target-modules`, adapter `g7-null-AE`.
  PASS: log "MoE model detected. Using fused MoE LoRA implementation." (lora/utils.py:102); no
  assert at fused_moe.py:39 or :77; KV ≥ 0.14 GiB (predicted ~0.7); null-adapter logprobs
  bitwise equal to base; then `g7-rand-E` changes them. FAIL modes to watch for: Triton
  `fused_moe_lora` compile error on sm_120, OOM at startup, a non-bitwise null result.
- **R5: a real trained adapter (G6 or an F3 rerun that saves).**
  Train/serve check on 20 held-out samples: NLL of base and adapter on the training stack
  (NF4, HF) and on the server (NVFP4/FP8, prompt_logprobs on the same token ids).
  PASS: same sign, and the served ΔNLL is ≥ 50% of the training-side ΔNLL. Measure A-only and
  A+experts separately. This is the plan's open "precision mismatch" UNKNOWN; see section 0
  for the per-module mismatch (attention matches; in_proj NF4 vs FP8; experts NF4 vs NVFP4).
- **Fallback if R4 fails: fold, then re-quantize.**
  - Tool (MEASURED `pip list`): **no nvidia-modelopt and no llm-compressor** in either
    `vllm/vllm-openai:v0.29.0` or `gpu-lab:training`. Both have **compressed-tensors 0.17.0**,
    which carries the NVFP4 building blocks: `pack_fp4_to_uint8`
    (compressors/nvfp4/helpers.py:34-75), `generate_gparam` (quantization/utils/helpers.py:309),
    the `NVFP4A16` preset (quantization/quant_scheme.py:145), and a ModelOpt->CT converter that
    renames `weight` to `weight_packed` without repacking (entrypoints/convert/converters/
    modelopt_nvfp4.py:45-46, 62-69), so the two formats share the nibble layout. vLLM ships a
    torch reference quantizer, `ref_nvfp4_quant` (quantization/utils/nvfp4_emulation_utils.py:414).
    No turnkey CLI: it is a short script that rewrites only the expert tensors in ModelOpt
    format (`weight` uint8, `weight_scale` e4m3 per 16, `weight_scale_2` fp32) and keeps
    hf_quant_config.json. W4A16 needs no calibration data. No new dependency (ARITHMETIC).
  - F0 (CPU): re-quantize the **BF16** checkpoint's experts with delta = 0 and compare with the
    NVFP4 checkpoint. PASS: packed bytes and `weight_scale` bitwise equal on ≥ 99.9% of blocks.
    That shows the recipe matches ModelOpt's; rounding and tie rules are UNKNOWN until then.
  - F1: W' = W_bf16 + `expert_lora.dense_delta` (expert_lora.py:131-138), re-quantize, and
    measure how much of the delta survives FP4:
    ‖deq(Q(W')) − deq(Q(W))‖ / ‖Δ‖ plus the share of changed codes. Risk (ARITHMETIC, not
    measured): a LoRA delta at lr 1e-4 may sit below one E2M1 step of most blocks and round
    away. Report this before trusting any served result.
  - F2: serve the folded checkpoint (unchanged shards symlinked) with the placement-A adapter
    via LoRA, then run R5.

## Also noted
- A second local image exists, `vllm/vllm-openai:nightly-dee37d89...` (MEASURED `docker images`);
  it was not read.
- The `try_get_optimal_moe_lora_config` tile lookup takes Marlin-packed weight shapes
  (punica_gpu.py:565-575; lora/layers/utils.py:104-138). That only affects tile configs, not
  correctness (inference from source).

# Evidence log: 4-bit LoRA of Nemotron 3.5 Lightning on one 24 GB GPU

Phase 0, "verify the negative". Run 2026-09-24, about 14:50–15:10 MDT, on the laptop
(RTX 5090 Laptop, sm_120) in `gpu-lab:training`, where MEASURED probes ran inside
throwaway `--rm` containers.

That image contains torch 2.13.0+cu130, transformers 5.16.1, peft 0.21.0 and
bitsandbytes 0.50.2.

Tags:
- **MEASURED**: a command run this session. The result JSON is in `results/`.
- **SOURCED**: a page or file actually opened. The URL is given.
  - *(raw)* means read with curl or from the installed package.
  - *(fetch)* means read through a summarising fetch tool. A *(fetch)* quote is weaker
    and should be re-read raw before anything depends on it.
- **ARITHMETIC**: the working is shown.
- **UNKNOWN**: followed by the experiment that would settle it.

## Verdicts

| # | Hypothesis | Verdict |
|---|---|---|
| H1 | NVIDIA's remote `modeling_nemotron_h.py` builds per-expert `nn.Linear`, so bnb quantizes the experts as-is | **EXISTS-UNTESTED**. The quantization step is MEASURED on the meta device: all 29.37B expert params become `Linear4bit`. Loading real weights, forward and training are untested. The code needs `mamba_ssm`. |
| H2 | A transformers release after 5.16.1 quantizes fused experts with bnb | **NOT FOUND** up to 5.17.0, the latest on PyPI (MEASURED: grep of the 5.17.0 wheel). |
| H3 | bitsandbytes after 0.50.2 supports 3D/MoE params | No newer release exists; `Experts4bit` (PR #1965) is still unmerged. **But 0.50.2 already ships `bitsandbytes.nn.parametrize.replace_parameter_4bit`**, and on transformers' own fused `NemotronHExperts` it is **EXISTS-WORKS at toy scale** (MEASURED). |
| H4 | Unsloth now trains MoE in 4-bit | **EXISTS-UNTESTED**. Open, unreleased PRs dated 2026-09-22..24 train `nemotron_h` remote-code models with 4-bit experts. The docs still say "not recommended". |
| H5 | Another stack does it | **axolotl `quantize_moe_experts`: EXISTS-UNTESTED** for nemotron_h. Its selection rule is architecture-agnostic, and it pins this image's exact stack. **experts4bit-qlora: EXISTS-UNTESTED**; Lightning loads and runs forward on CPU, but the author claims no training support for nemotron_h. ms-swift, LLaMA-Factory, torchao NF4, HQQ and NeMo AutoModel: NOT FOUND (details below). |
| H6 | PEFT `target_parameters` works on quantized experts | **NOT FOUND in stock PEFT 0.21**, which refuses (MEASURED: `NoMatchingPeftModuleError`). The bf16 control works, so the parametrization is the cause. **EXISTS-UNTESTED** through axolotl's patch (SOURCED). |

The previous session's premise, that 4-bit fused-expert training has to be written
from scratch, is wrong twice over. NVIDIA's own code sidesteps the fused layout, and
bnb 0.50.2 has a primitive that quantizes the fused layout.

## H1: NVIDIA remote code

- **SOURCED (raw)**: of the three repos checked, `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
  (rev `bf77c317`) and `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (rev `2dc98e2a`)
  ship `modeling_nemotron_h.py` and `configuration_nemotron_h.py`, and
  `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (rev `a9904d24`) ships neither.
  Evidence: the HF API file listing, `https://huggingface.co/api/models/<repo>`.
  - sha256 of Nano's modeling file:
    `4d353ce6e8f495d043f2d8c5acd13496a84ba2b2d45da8279797a55f680309d9`
  - sha256 of Super's modeling file:
    `e1cb5fc02e887983f0a445bf4c1a2604453b2cb2db4624c7004dcf663bbb1b6e`
- **SOURCED (raw)**, both files: `NemotronHMOE.experts = nn.ModuleList([NemotronHMLP(...)
  for _ in range(n_routed_experts)])`. Each `NemotronHMLP` holds two plain `nn.Linear`
  (`up_proj`, `down_proj`) with relu2 between them.
- **SOURCED (raw)**: Lightning's `config.json` equals Nano's on every shared key except
  `transformers_version` (4.57.6 vs 4.55.4).
  - Lightning's `layers_block_type` is identical to Nano's expanded `hybrid_override_pattern`
    (`MEMEM*EMEM…`, 52 layers).
  - Lightning adds `moe_latent_size: null`, `moe_shared_expert_overlap`,
    `mtp_layers_block_type` and `num_nextn_predict_layers: 1`.
  - Super's `configuration_nemotron_h.py` reads all of those natively.
- **SOURCED (raw)**: Lightning's safetensors index contains every one of Nano's 6243
  tensor names, plus 270 `mtp.*` tensors. Super's code sets
  `_keys_to_ignore_on_load_unexpected = [r"mtp.*"]` and `_checkpoint_conversion_mapping
  = {"backbone": "model"}`. Nano's code uses `backbone.` directly.
- **MEASURED**, `results/h1-{super,nano}-remote-code-meta.json`, probe
  `probes/h1_remote_code_bnb_count.py`:
  - Method: Lightning's config plus NVIDIA code, built on the meta device, then the same
    `replace_with_bnb_linear` call as the original blocker measurement.
  - Both code versions give 31.58B params total:
    - **30.86B** in `Linear4bit` (6004 modules)
    - **29,374,808,064 expert params in `Linear4bit` and 0 expert params left bf16**
    - 0.71B not 4-bit: embedding 352M, `lm_head` 352M, router 7.9M, conv1d, norms
  - For contrast, the blocker (the built-in class) left 29.38B in bf16.
- **ARITHMETIC**: 23 MoE layers × 128 experts × 2 matrices × 2688 × 1856 = 29,374,808,064,
  which matches the measured count exactly.
- **Caveats**:
  - The run was meta-device only: nothing was allocated and no real weights were loaded.
  - A stub stood in for `mamba_ssm`. **Both NVIDIA files `raise ImportError` if
    `mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn` is missing** (SOURCED (raw): Nano
    lines 62–66, Super lines 60–63), so a real run needs `mamba_ssm` or a patch.
  - The code was written for transformers 4.55. Only import and construction are proven
    on 5.16.1.
  - Forward, generate and gradient checkpointing under 5.16.1 are UNKNOWN. Settle them
    with a real-weight load plus one training step (G4/G5).
- **Cost risk, UNKNOWN until G5**:
  - The MoE forward is a Python loop over all 128 experts in each of 23 layers.
  - Every idle expert runs a dummy one-token forward, `zeros(...).to(expert.down_proj.weight.dtype)`.
    Under bnb that dtype is uint8.
  - Settle it by measuring tokens/s against route 1 below.
- **Known bugs on this route** (SOURCED (fetch), unsloth-zoo PR #1340, open, found on
  NVIDIA's Nemotron-Labs-Teacher remote code):
  1. relu2 uses `torch.square`, which autocast promotes to fp32, so `index_add_` fails with
     "self (BFloat16) and source (Float) must have the same scalar type".
  2. The uint8 dummy input reaches PEFT's 4-bit LoRA matmul: "expected mat1 and mat2 to
     have the same dtype, but got: unsigned char != c10::BFloat16".

  Both fixes are small (`y*y`, and a cast before the LoRA matmul).
- Not checked: `transformers` v4.57.6 and v4.57.0 have no file at
  `src/transformers/models/nemotron_h/modeling_nemotron_h.py` (HTTP 404, raw GitHub).
  I did not look for another path, because NVIDIA's own code answered the question.

## H2: transformers after 5.16.1

- **SOURCED (raw)**: PyPI lists `transformers` 5.17.0 (2026-09-09) as the latest release.
  Releases since Aug 1: 5.15.0, 5.15.1, 5.16.0, 5.16.1, 5.17.0.
- **MEASURED** (the whole 5.17.0 wheel was unpacked and grepped):
  - `integrations/bitsandbytes.py` and `quantizers/quantizer_bnb_4bit.py` have no expert
    or 3D handling.
  - No file anywhere in the wheel mentions `replace_parameter_4bit` or `Experts4bit`.
    `Params4bit` appears only in the bnb integration, in generic `modeling_utils` code and
    in RWKV.
  - `NemotronHExperts` is still `@use_experts_implementation(has_gate=False)` with fused
    3D `nn.Parameter`s.
- **SOURCED (fetch)**: bnb issue #1849, "Failed to quant MoE models with fused expert
  weights in transformers v5", opened 2026-01-25, is still open with no maintainer comment.
  It points to transformers #43472, a `BatchLinear` proposal; I did not open that one.

## H3: bitsandbytes

- **SOURCED (raw)**: PyPI lists bitsandbytes 0.50.2 (2026-08-27) as the latest.
  The releases page (fetch) mentions no MoE or experts through 0.50.2.
- **SOURCED (fetch)**: PR #1965, "Add Experts4bit" by pjordanandrsn:
  - opened 2026-06-05, still open on 2026-09-19;
  - a maintainer said on 2026-06-10 it would not merge before 0.50.0, and on 2026-08-08
    asked for CI;
  - tested on an RTX A2000.
- **SOURCED (raw, installed package)**: bnb 0.50.2 has `bitsandbytes/nn/parametrize.py`
  with `replace_parameter_4bit(module, param_name, compress_statistics, quant_type,
  blocksize)`.
  - It quantizes any `nn.Parameter` with `F.quantize_4bit`.
  - It registers a `torch.nn.utils.parametrize` hook that dequantizes whenever the
    parameter is accessed.
  - Its docstring says it is "useful for MoE models or other scenarios where you want to
    quantize parameters outside of nn.Linear layers without changing the model's
    architecture", marked "experimental".
- **MEASURED**, `results/h3-bnb-parametrize-tiny.json`, probe
  `probes/h3_bnb_parametrize_tiny.py`:
  - Model: a tiny random `nemotron_h` built from the built-in transformers class
    (hidden 256, layers mamba/moe/attention/moe, 8 experts top-2), on the RTX 5090 Laptop.
  - Experts implementation: `grouped_mm`.
  - Expert data 2,097,152 B → 524,288 B, packed uint8. That is exactly 4×; the absmax
    statistics are held separately and are not included in the count.
  - Logits, 4-bit model vs a dense twin holding the dequantized weights:
    **relative difference 0.0**.
  - Gradient at layer 0, which has to pass back through both 4-bit MoE layers:
    **relative difference 0.0** vs the dense twin, and finite.
  - Standard PEFT LoRA on q/k/v/o, in_proj and shared_experts up/down, with gradient
    checkpointing, 8 AdamW steps: loss 6.986 → 6.667, decreasing monotonically.
  - The experts were still uint8 afterwards.
  - Relative logit error vs the original bf16 weights is 0.139. This is **NF4 error on
    random-init toy weights and not representative**; G2 measures it on real layers.
- **UNKNOWN at real size**: peak VRAM and speed. The parametrization dequantizes a layer's
  entire 3D expert stack each time it is accessed. Per layer that transient is
  - **ARITHMETIC**: 128 × 2 × 2688 × 1856 × 2 B = 2.55 GB (2.38 GiB) in bf16.

  Whether autograd keeps that copy for backward, and how much of it gradient
  checkpointing avoids, is settled only by G5's `max_memory_allocated`.

## H4: Unsloth

- **SOURCED (fetch)**: the page `https://unsloth.ai/docs/basics/faster-moe` today still
  reads "Training MoE models in 4-bit QLoRA isn't recommended right now because
  BitsandBytes doesn't support it", and its supported MoE list does not include Nemotron.
- **SOURCED (fetch)**: the page `https://unsloth.ai/docs/models/nemotron-3.5` says fine-tuning
  is supported ("the entire NVIDIA Nemotron model family"). It states no VRAM figure and
  no 4-bit statement.
- **SOURCED (fetch)**, open PRs, none merged or released. Their states and creation
  dates were re-checked raw through the GitHub REST API on 2026-09-24:
  - unsloth #11543 (2026-09-22): per-expert LoRA on `nemotron_h` remote-code
    `mixer.experts.<i>.up_proj`, 4-bit. Tested on Nemotron-Labs-Teacher (512 experts) on
    3× B200: "5.72 B LoRA parameters, 5.67 B of them on the experts", losses 4.53 → 1.79
    over 10 steps.
    - It notes that bnb quantizing `out_proj` "causing kernel failures", worked around by
      quantizing selectively.
  - unsloth #11526 (2026-09-22..24): 4-bit and 16-bit training of Nemotron-3-Nano-Omni
    (a 30B-A3B `nemotron_h` backbone) completed; the GPU and VRAM were not in the summary.
  - unsloth-zoo #1340: the two bugs listed under H1.
  - unsloth-zoo #1348 and #1336 (2026-09-22): generic routing of bnb 4-bit experts for
    other families. Nemotron is not named.
  - unsloth-zoo issue #850 (closed): `load_in_4bit` quantized fused experts for
    architectures whose forward was not patched, which then crashed at the first forward.
- Hardware for a 4-bit Lightning run on 24 GB: **UNKNOWN**. No Unsloth source gives one.

## H5: other stacks

- **axolotl**:
  - **SOURCED (raw)**: `src/axolotl/monkeypatch/moe_quant.py` on main. It patches
    `transformers.core_model_loading.set_param_for_module`. Every parameter that has
    `ndim >= 3`, is on CUDA, has **"expert" in its name** and is not already bnb gets
    `bitsandbytes.nn.parametrize.replace_parameter_4bit(..., quant_type="nf4",
    compress_statistics=True)`, after which the bf16 tensor is freed.
    - It also turns off `caching_allocator_warmup`, which pre-allocates at bf16 size.
    - Lightning's fused `…mixer.experts.up_proj` / `down_proj` match this rule;
      `conv1d.weight` (3D, no "expert") is skipped.
  - **SOURCED (raw)**: axolotl 0.19.0 (PyPI, 2026-09-10) pins `transformers==5.16.1`,
    `bitsandbytes==0.50.2`, `torch>=2.11,<=2.13.0` and `peft==0.20.0`. That is this
    image's stack except PEFT (the image has 0.21.0).
  - **SOURCED (fetch)**, docs `expert_quantization.html`:
    - Enabled with `quantize_moe_experts: true` plus `adapter: qlora` and `load_in_4bit`.
    - Expert LoRA goes through `lora_target_parameters` and needs `lora_dropout: 0`.
    - "GLM-4.7-Flash QLoRA drops from ~127GiB to ~23GiB reserved memory".
    - Nemotron is not listed as tested.
  - **SOURCED (fetch; merge date raw via GitHub API)**: PR #3866, merged 2026-09-16, adds Nemotron-3 latent-MoE
    non-gated relu2 experts to ScatterMoE and SonicMoE, with NVFP4 checkpoint loading.
    Its example targets Super-120B-NVFP4. Per the `nvfp4_lora` docs, SonicMoE does not
    support consumer Blackwell sm_120, while ScatterMoE works on sm80+.
  - Verdict: **EXISTS-UNTESTED** for Lightning.
- **experts4bit-qlora** (third party, MIT, pjordanandrsn; PyPI 0.37.3, 2026-09-24; 71
  releases since 2026-07-01):
  - **SOURCED (raw)**: `docs/ARCHITECTURE_SUPPORT.md` at main `e459ad99ab` has the row
    `nemotron_h | nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | reference | cpu | ok |
    23q/0u | ok | reference-ok`. That is load plus quantize-verify plus forward, **on CPU**.
  - **SOURCED (raw)**: `docs/capabilities.json` has no `nemotron_h` in any `model_families`
    list. Training support is claimed only for olmoe, qwen3_moe, gemma4_text, mixtral and
    granitemoe.
  - Verdict: **EXISTS-UNTESTED**. It quantizes Lightning; training it is outside the
    author's own claims.
- **ms-swift**:
  - **SOURCED (raw)**: `swift/model/models/nvidia.py` on main registers
    `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/-NVFP4` with `default_trust_remote_code =
    False`, and requires `transformers>=5.0`, `mamba-ssm` and `causal-conv1d>=1.2.0`.
  - It therefore uses the built-in fused class, which the original blocker measurement
    showed leaves 29.38B params in bf16 under bnb. None of the files read show separate
    expert quantization.
  - Verdict: **NOT FOUND** for 4-bit experts.
- **LLaMA-Factory**: **SOURCED (raw)**: `src/llamafactory/extras/constants.py` on main
  (3721 lines) contains no "nemotron". **NOT FOUND**.
- **torchao**: **SOURCED (fetch)**: the releases page shows MoE training work only for
  MXFP8. Nothing says NF4Tensor handles 3D expert weights. **NOT FOUND**.
  - A search snippet says `Int4Tensor` accepts 3D expert stacks; RFC #4939 was not opened.
- **HQQ**: **SOURCED (fetch)**: transformers `docs/.../quantization/hqq.md` says it
  quantizes "all the linear layers (torch.nn.Linear)". Nothing covers fused 3D experts.
  **NOT FOUND**.
- **NeMo AutoModel**:
  - **SOURCED (fetch)**: the release notes to 0.5.0 mention MoE LoRA (0.4.0) and QLoRA
    checkpoints, but no QLoRA with quantized routed experts.
  - **SOURCED (fetch)**: an NVIDIA forum post (2026-06-05) fine-tuned Nano-30B-A3B on a
    GB10 with NeMo AutoModel 0.3.0 `force_hf`, transformers 5.5.0, "4-bit QLoRA", and
    reported "Load then peaks ~20 GB, training ~35 GB".
    - **ARITHMETIC**: bf16 experts alone are 29.37B × 2 B = 58.7 GB, so a 20 GB load peak
      is *consistent with* 4-bit experts.
    - The post does not say so, and its code is in PDFs I did not open.
  - Verdict: **EXISTS-UNTESTED**, ambiguous.

## H6: PEFT

- **MEASURED**, `results/h6-peft-target-parameters-tiny.json` and `…-control-bf16.json`,
  probe `probes/h6_peft_target_parameters_tiny.py`:
  - With `LoraConfig(target_parameters=["experts.up_proj","experts.down_proj"])` on the
    tiny model with bnb-parametrized 4-bit experts, stock PEFT 0.21 raises
    `NoMatchingPeftModuleError: No target_modules passed but also no target_parameters
    found`.
  - The same config on bf16 experts injects, gives the expert LoRA non-zero gradients, and
    lowers the loss (6.985 → 6.720 over 6 steps).
- **SOURCED (raw)**: axolotl's `patch_peft_target_parameters_matching()` exists to "Fix
  PEFT's _inject_parameters for target_parameters on quantized MoE experts", by expanding
  short suffixes to full module paths for parametrized modules. Whether it works here is
  **UNKNOWN**; settle it by applying the patch in the h6 probe.
- **MEASURED**, found in passing: PEFT 0.21 refuses LoRA on Mamba `out_proj` and `conv1d`
  for `model_type='nemotron_h'`: "Module 'out_proj' is incompatible with Mamba-based
  models". **The hand-off's placement A listed mamba `out_proj`, so it must drop it.**

## What this changes for Phase 1

Two routes now reach 4-bit experts. Neither needs custom autograd, so the hand-off's G3
(a custom `autograd.Function` plus gradcheck) is replaced by measurement.

1. **Built-in class + `replace_parameter_4bit` on the fused experts**, the mechanism behind
   axolotl's `quantize_moe_experts`.
   - For: it works at toy scale in this image (MEASURED), keeps the `grouped_mm` expert
     path, and adds no dependency.
   - Open questions:
     - Loading 61.3 GiB of bf16 while never holding it all at once needs a load-time hook
       (axolotl's `patch_moe_quantization_on_load` is ~70 lines) or axolotl itself.
     - Expert LoRA (placement B) needs a PEFT fix.
     - The per-layer dequantize transient is 2.55 GB.
2. **NVIDIA remote code + standard `BitsAndBytesConfig`**.
   - For: stock bnb quantizes all experts (MEASURED on meta); per-expert `Linear4bit` gives
     standard PEFT targets for placement B.
   - UNKNOWN until G4: whether the standard `from_pretrained` + `BitsAndBytesConfig` load
     stays under 64 GB of RAM and 24 GB of VRAM for this checkpoint.
   - Costs:
     - it needs `mamba_ssm` built for sm_120 and sm_86;
     - the two small bug fixes under H1;
     - the MoE forward is a Python loop with dummy compute on idle experts.

Either way, G0 comes first (the 61.3 GiB download). G4 (idle VRAM) and G5 (peak VRAM,
tokens/s) then decide between the routes on this hardware.

## Could not verify

- Every real-size memory and speed figure for both routes. That is G4/G5, and by rule
  these stay UNKNOWN until measured.
- NVIDIA remote code forward/backward under transformers 5.16.1, which needs `mamba_ssm`.
- axolotl `quantize_moe_experts` on nemotron_h end to end, and its PEFT patch.
- Unsloth's PR branches, which are unreleased and were not installed.
- The code in the NVIDIA forum post's PDFs. It is not established whether the routed
  experts were 4-bit there.
- *(fetch)*-sourced quotes are summaries from a fetch tool and have not been re-read raw.
  Any decision that rests on one should re-open it first.

## Search trail (all opened 2026-09-24)

Raw (curl, or the installed package):
- https://huggingface.co/api/models?author=nvidia&search=Nemotron-3&limit=100
- https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (and -3-Super-120B-A12B-BF16, -3.5-Lightning-30B-A3B-BF16)
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/{modeling_nemotron_h.py,configuration_nemotron_h.py,config.json,model.safetensors.index.json}
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/resolve/2dc98e2afe4face0e4ce40972a915c45368bd34a/{modeling_nemotron_h.py,configuration_nemotron_h.py,config.json}
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/resolve/a9904d24bcc1d289a1950fa9d2b978c47cf903b9/{config.json,model.safetensors.index.json}
- https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/models/nemotron_h/modeling_nemotron_h.py (404; v4.57.0 also 404)
- https://api.github.com/repos/huggingface/transformers/commits?path=src/transformers/models/nemotron_h/modeling_nemotron_h.py
- https://pypi.org/pypi/{transformers,bitsandbytes,peft,axolotl,experts4bit-qlora}/json; transformers-5.17.0-py3-none-any.whl
- https://raw.githubusercontent.com/pjordanandrsn/experts4bit-qlora/main/docs/{ARCHITECTURE_SUPPORT.md,capabilities.json,STATUS.md}
- https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/src/axolotl/monkeypatch/moe_quant.py
- https://raw.githubusercontent.com/hiyouga/LLaMA-Factory/main/src/llamafactory/extras/constants.py
- https://raw.githubusercontent.com/modelscope/ms-swift/main/swift/model/constant.py and swift/model/models/nvidia.py
- In `gpu-lab:training`: `bitsandbytes/nn/parametrize.py`, `peft/tuners/lora/layer.py` (ParamWrapper), `transformers/models/nemotron_h/modeling_nemotron_h.py`

Fetch (summarised):
- https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1965
- https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1849
- https://github.com/bitsandbytes-foundation/bitsandbytes/releases
- https://pypi.org/project/experts4bit-qlora/
- https://github.com/axolotl-ai-cloud/axolotl/pull/3866
- https://github.com/axolotl-ai-cloud/axolotl/releases
- https://docs.axolotl.ai/docs/nvfp4_lora.html
- https://docs.axolotl.ai/docs/expert_quantization.html
- https://docs.axolotl.ai/docs/api/monkeypatch.moe_quant.html
- https://unsloth.ai/docs/basics/faster-moe
- https://unsloth.ai/docs/models/nemotron-3.5
- https://github.com/unslothai/unsloth/pull/11543
- https://github.com/unslothai/unsloth/pull/11526
- https://github.com/unslothai/unsloth-zoo/pull/1336
- https://github.com/unslothai/unsloth-zoo/pull/1340
- https://github.com/unslothai/unsloth-zoo/pull/1348
- https://github.com/unslothai/unsloth-zoo/issues/850
- https://forums.developer.nvidia.com/t/fine-tuning-nemotron-3-nano-30b-a3b-on-asus-ascent-gx10-gb10-dgx-spark-cuda-oom-at-load-box-freezes-4-bit-qlora-fix/372434
- https://docs.nvidia.com/nemo/automodel/whats-new/release-notes
- https://github.com/pytorch/ao/releases
- https://github.com/huggingface/transformers/blob/main/docs/source/en/quantization/hqq.md

Failed: https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/lora.md
(404; replaced by reading the installed PEFT source).

Web searches run:
- "transformers v5 bitsandbytes 4-bit MoE fused experts quantization support"
- "bitsandbytes release notes 2026 MoE experts 3D parameter 4bit"
- "unsloth Nemotron 3.5 Lightning fine-tune VRAM QLoRA"
- "axolotl QLoRA MoE experts 4-bit ScatterMoE kernels nemotron"
- "PEFT target_parameters LoRA quantized bitsandbytes MoE experts"
- "ms-swift Nemotron-3 Nano QLoRA fine-tuning"
- "huggingface transformers pull request bitsandbytes quantize fused MoE experts 3D parameter 2026"
- "NeMo Automodel QLoRA Nemotron 3 Nano 30B-A3B 4-bit experts peft"
- "LLaMA-Factory Nemotron-3 Nano nemotron_h QLoRA support"
- "torchao NF4Tensor 3D tensor MoE experts QLoRA support"
- "HQQ quantization MoE fused experts transformers v5 training LoRA"
- "unsloth nemotron_h 4-bit experts load_in_4bit NemotronHExperts"
- "axolotl quantize_moe_experts docs supported models bnb 4-bit experts"

Leads seen but not opened:
- woct0rdho/transformers-qwen3-moe-fused
- torchao RFC #4939
- axolotl PR #4003
- transformers issue #43472
- unsloth PR #5432
- third-party blogs (DataCamp, AceCloud, MindStudio)

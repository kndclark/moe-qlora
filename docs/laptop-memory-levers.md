# Laptop memory: what frees VRAM and KV on one card

Date: 2026-10-05. David asked whether any tool, formula ("Octization"?) or OS change makes
serving and training on the laptop alone use VRAM, RAM, tokens and storage more efficiently,
and whether the KV cache can grow. This note answers with measurements where a GPU run could
settle it, and with arithmetic or sources where it could not.

Scope: the laptop RTX 5090 Laptop (23.4 GiB), vLLM 0.29.0 (`vllm/vllm-openai:v0.29.0`),
Lightning NVFP4 (revision `bee7596`) with the G6q adapter. These are the flags of every eval
server in `probes/` (`--kv-cache-dtype fp8 --moe-backend marlin --max-model-len 16384
--max-num-seqs 16`) unless a row says otherwise. Eval settings: v1, v2, alert and trap3,
thinking on, 4,096 max tokens, concurrency 16, seed 20260923.

Label key: **[measured]** observed on our files or hardware; **[sourced]** from a cited
source; **[arithmetic]** computed here; **[inference]** judgement, not tested.

## Bottom line

1. **The eval server we use queues 9 of its 16 requests for lack of KV, and one flag fixes
   it.**
   - The G6q eval server ran with 0.45 GiB of KV: room for 3.08 requests of 16k tokens. It
     peaked at 7 running and 9 waiting, with the KV cache 97.2% full [measured,
     `results/g6q-eval-serve.log`].
   - The cause is `--enable-lora`. It attaches a rank-16 adapter slot to every linear layer
     it can, including the 23 × 128 routed experts. G6q never targets those, and the slots
     take 0.82 GiB ("Model loading took" 18.70 vs 17.88 GiB) [measured]. The arithmetic
     agrees: 2 matrices × (16 × 2,688 + 1,856 × 16) parameters × 2 B × 2,944 experts =
     0.80 GiB [arithmetic].
   - The fix: `--lora-target-modules q_proj k_proj v_proj o_proj in_proj up_proj down_proj`
     ("tm" below) frees 1.0 GiB. KV goes from 0.45 to 1.45 GiB (3.2x), and the server runs
     all 16 requests with 0 waiting [measured].
   - The adapter is still applied: tm's answers match G6q's (v2 agreement 129–132 of 134)
     and not base Lightning's (92–97) ("How the answers were checked") [measured].
2. **With tm, CUDA graphs fit at `--gpu-memory-utilization 0.92` and the eval runs about 3x
   faster.**
   - Graphs do not fit at 0.85 ("0.14 GiB KV cache is needed … (0.04 GiB)") [measured].
   - At 0.92, each split's wall time drops against the reference runs [measured]:

     | Split | Reference runs | tm + graphs |
     |---|---|---|
     | v1 | 100–110 s | 33–37 s |
     | v2 | 95–102 s | 30–34 s |
     | alert | 17 s | 4–6 s |
     | trap3 | 10–11 s | 3–4 s |

   - About 1.4x of that is graphs themselves: peak aggregate decode is 266 tok/s eager and
     355–396 tok/s with graphs, both with tm [measured]. The rest is no longer queueing.
3. **On this hybrid model, the cheapest KV gain comes from the Mamba state, not from KV
   bytes.**
   - Attention KV is 3,072 B per token at fp8, because only 6 of 52 layers are attention
     and each has 2 KV heads [arithmetic, config.json]. A 16k request's pages are mostly
     Mamba state.
   - `--mamba-ssm-cache-dtype float16` halves the state page: 1.47x the 16k concurrency.
   - `--mamba-cache-mode none` drops the per-request prefix-cache copies: 1.50x.
   - Both together give 2.03x, from 11.75x to 23.83x [measured].
   - No net change in scores. fp16 was repeated three times: v2 signed sums +112, +116 and
     +114 against references of 112, 115 and 116 [measured].
   - The state path does flip one borderline item each way: `two_flag-chronyc` down and
     `task-nft-13` up, in about half the servers that change it. Of the four servers that
     keep it, only the expert-offload one flips either item ("How the answers were
     checked").
   - A CPU simulation of the state recurrence passes fp16 at 4,096 steps, but the error
     keeps growing past that. So fp16 is cleared for these ≤4k-token traces, not for long
     contexts (see "State formats").
4. **vLLM overestimates CUDA-graph memory 7x and takes the guess out of the KV budget.**
   - The logs say "CUDA graph pool memory: 0.09 GiB (actual), 0.65 GiB (estimated)", and
     the estimate is subtracted from KV [measured, log; sourced, vLLM
     `v1/worker/gpu_worker.py`].
   - With `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` on top of fp16 state and `none`
     mode, KV goes from 1.74 to 2.34 GiB (+0.60 GiB, the estimate's size) [measured].
   - 16k concurrency goes from 23.83x to 32.00x: 10.4 times the committed server's 3.08x,
     with graphs on [measured].
   - `--kv-cache-memory 2684354560` (2.5 GiB) also runs, at 34.17x. But it skips memory
     profiling altogether, so its margin is unmeasured. The environment variable keeps
     the measured peak and drops only the guess [measured, log line "skipped memory
     profiling"].
5. **No new number format pays on this model.**
   - Experts cannot be stored as deltas from each other: they are as far apart as unrelated
     matrices (distance 1.28–1.42 of their own norm; √2 = 1.41 would be fully orthogonal)
     [measured].
   - A cheaper shared-exponent weight format saves about 7% at the cost of a new kernel
     [arithmetic].
   - KV formats (TurboQuant, int4) shrink a term that is already tiny here [arithmetic].
   - The working RAM-for-VRAM trade is expert offload, and on the eval it costs more
     than it buys. `--cpu-offload-gb 4 --cpu-offload-params experts` moves 3.95 GiB of
     experts to RAM. KV goes from 1.69 to 5.63 GiB (39.25x), but v1 + v2 take 259 s
     against 63–71 s on the card, and peak decode falls from 355–396 to 116 tok/s
     [measured, row 14]. That is slower than the committed server (195–207 s), which
     queues 9 requests. Offload is worth it only when a context or batch does not fit
     at all.
6. **A different Linux would not free VRAM.** The one OS-level lever is getting the display
   off the NVIDIA card: 0.2–1.2 GiB, via a BIOS change David has to make. It is untested,
   and the BIOS "iGPU only" mode must be avoided because it hides the dGPU from Linux
   (see "OS and system").

## Where the bytes go

### The budget

vLLM requests `utilization × 23.4 GiB`. From that it subtracts weights plus non-torch memory
(CUDA context and workspaces) and the peak activation measured in a profiling pass. What
remains is the KV pool [sourced, `gpu_worker.py` `determine_available_memory`; measured,
each row reproduces from its log line].

| Config | Requested | Model loading | Consumed (weights + non-torch) | Peak act. | KV |
|---|---|---|---|---|---|
| Committed: full LoRA, eager, 0.85 | 19.89 | 18.70 | 19.26 | 0.18 | 0.45 |
| tm, eager, 0.85 | 19.89 | 17.88 | 18.26 | 0.18 | 1.45 |
| tm, eager, 0.92 | 21.53 | 17.88 | 18.26 | 0.18 | 3.09 |
| tm, graphs, 0.92 | 21.53 | 17.88 | 18.70 | 1.13 | 1.69 |
| No LoRA, graphs, 0.92 | 21.53 | 17.86 | 18.68 | 1.06 | 1.79 |

All figures are GiB [measured, `results/kv-levers/serve-*.log`].

- **tm frees 1.00 GiB.** That is 0.82 GiB of expert adapter slots plus 0.18 GiB of non-torch
  memory (0.56 GiB with full LoRA, 0.38 with tm).
- **Graphs cost 1.40 GiB of KV at the same utilization:** 3.09 GiB eager vs 1.69 GiB with
  graphs. The cost has three parts:
  - **The graph estimate, 0.65 GiB** (0.59 without LoRA), counted inside "peak activation".
  - **A larger profiling transient, 0.30 GiB:** 0.47–0.48 GiB with graphs vs 0.18 eager. It
    is the same with and without LoRA, so it is graphs, not the adapter.
  - **0.44 GiB more non-torch memory:** consumed minus model loading is 0.82 GiB with graphs
    vs 0.38 eager. The cause is unknown [measured; inference that it is compiled kernels].
- **LoRA itself, once tm is set, costs 0.1 GiB of KV** in graph mode (1.69 vs 1.79)
  [measured].

### Why the 16k concurrency moves in steps

Lightning is 23 Mamba layers, 23 MoE layers and 6 attention layers [config.json]. vLLM pages
a hybrid model as follows [measured, logs; arithmetic below]:

- **One page per layer group.** A page holds one block of a 6-layer group: one attention
  group and four Mamba groups (23 Mamba layers plus 1 padding layer).
- **Pages are sized to the Mamba state, and the attention block grows to fill them.** A
  Mamba layer's state is 64 heads × 64 × 128 × 4 B + conv 6,144 × 3 × 2 B = 2.03 MiB at
  fp32. At fp8, one attention layer stores 512 B per token, so the attention block is
  4,176 tokens. The page is 12.24 MiB [arithmetic; the log says "Setting attention block
  size to 4176"].
- **fp16 state halves the page:** 6.23 MiB, with a 2,128-token attention block [measured,
  log].
- **Blocks per 16k request** = ceil(16,384 / attention block) + 4 Mamba groups × k. With
  prefix caching (`align` mode) k = 2; with `none` mode k = 1.

| Config | Blocks per 16k request |
|---|---|
| fp32 state, align | 12 |
| fp32 state, none | 8 |
| fp16 state, align | 16 |
| fp16 state, none | 12 |

Concurrency = floor(KV / page) / blocks per request. This reproduces every concurrency vLLM
printed [arithmetic vs measured]:

| Run | Blocks | Arithmetic | Printed |
|---|---|---|---|
| Committed | 37 | 37 / 12 = 3.08 | 3.08x |
| tm eager 0.85 | 121 | 121 / 12 = 10.08 | 10.08x |
| tm eager 0.92 | 258 | 258 / 12 = 21.50 | 21.50x |
| tm graphs 0.92 | 141 | 141 / 12 = 11.75 | 11.75x |
| + fp16 state | 277 | 277 / 16 = 17.31 | 17.31x |
| + `none` mode | 141 | 141 / 8 = 17.62 | 17.62x |
| + fp16 + `none` | 286 | 286 / 12 = 23.83 | 23.83x |
| No LoRA, graphs 0.92 | 149 | 149 / 12 = 12.42 | 12.42x |

Two consequences [inference]:

- **The Mamba state is the binding term.** At fp32, 8 of the 12 blocks in a 16k request are
  state. KV formats leave it untouched, and when they shrink attention bytes, vLLM grows
  the attention block to keep pages state-sized.
- **The 16k figure is a worst case.** Mamba blocks are taken at admission, attention blocks
  as tokens arrive. Our requests are under 5.2k tokens (prompt ≤ 1,046 plus 4,096 think),
  so they need fewer attention blocks than the 16k figure assumes. That is why 1.45 GiB
  already runs all 16 requests at once.

### The graph-memory estimate

In vLLM 0.29.0, `determine_available_memory` profiles a graph capture and adds its estimate
to peak activation. It is controlled by `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`, default 1
[sourced, `envs.py`]. The log then compares the estimate with the real pool [measured]:

| Run | Estimated | Actual |
|---|---|---|
| tm | 0.65 GiB | 0.09 GiB |
| tm, fp16 + `none` | 0.60 GiB | 0.09 GiB |
| No LoRA | 0.59 GiB | 0.08 GiB |

The overestimate is 0.51–0.56 GiB of KV that is never used. The engine itself suggests
2.73 GiB to "fully utilize gpu memory" in the tm graphs run [measured,
`serve-tm-graphs092.log`]. Two ways around it were tested, both with fp16 state and `none`
mode [measured, rows 12 and 13 below]:

| Run | How KV is sized | Peak act. | KV | 16k conc. |
|---|---|---|---|---|
| tm, fp16 + `none`, 0.92 | profiled, estimate applied | 1.08 | 1.74 GiB | 23.83x |
| tm-noest: same + `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` | profiled, no estimate | 0.48 | 2.34 GiB | 32.00x |
| tm-stack: same + `--kv-cache-memory 2684354560`, 0.85 | fixed 2.5 GiB, no profiling | not measured | 2.50 GiB | 34.17x |

- **noest still runs a profiling capture and logs the same 0.65 GiB guess.** It only leaves
  the guess out of the budget, so the measured transient (0.48 GiB) still protects the run
  [measured, `serve-tm-noest.log`].
- **`--kv-cache-memory` logs "reserved 2.5 GiB … and skipped memory profiling".** It ignores
  `--gpu-memory-utilization`, so nothing checks that 2.5 GiB fits. It did fit, with 22.88 GiB
  free at start [measured, `serve-tm-stack.log`]. A window opened later would not be
  accounted for [inference].
- **Choose noest:** 0.16 GiB less than the stack, with its margin measured rather than
  assumed [inference].

## Measured levers

Each row is one server start followed by v1, v2, alert and trap3.

- **Answers columns** give the signed headline sum (scored items) for v2 and alert. The
  three reference runs score 112/115/116 on v2 and 7/3/5 on alert. Across rows, v1 scores
  127–129 (refs 128–129) and trap3 4–6 (refs 5 each time).
- **Wall** is v1 + v2, from each eval's `elapsed_s`.

| # | Config | KV GiB | 16k conc. | Running/waiting (peak) | v2 | alert | Wall |
|---|---|---|---|---|---|---|---|
| 1 | Committed (full LoRA, eager, 0.85) | 0.45 | 3.08x | 7 / 9 | refs | refs | 195–207 s |
| 2 | tm, eager, 0.85 | 1.45 | 10.08x | 16 / 0 | +115 | +4 | 282 s* |
| 3 | tm, graphs, 0.85 | fails | — | — | — | — | — |
| 4 | tm, graphs, 0.92 | 1.69 | 11.75x | 16 / 0 | +116 | +5 | 65 s |
| 5 | + fp16 state (3 runs) | 1.69 | 17.31x | 16 / 0 | +112/+116/+114 | +5 ×3 | 64–66 s |
| 6 | + `none` mode (fp32 state) | 1.69 | 17.62x | 16 / 0 | +112 | +4 | 68 s |
| 7 | + fp16 + `none` | 1.74 | 23.83x | 16 / 0 | +116 | +5 | 66 s |
| 8 | b512 (`--max-num-batched-tokens 512`), 0.92 | 1.61 | 11.17x | 15 / 0 | +114 | +6 | 71 s |
| 9 | b512, 0.85 | fails | — | — | — | — | — |
| 10 | tm, eager, 0.92 (capacity only) | 3.09 | 21.50x | — | — | — | — |
| 11 | No LoRA, graphs, 0.92 (capacity only) | 1.79 | 12.42x | — | — | — | — |
| 12 | Row 7, KV fixed at 2.5 GiB (`--kv-cache-memory`) | 2.50 | 34.17x | 16 / 0 | +118 | +3 | 63 s |
| 13 | Row 7 + graph estimate off (ESTCG=0) | 2.34 | 32.00x | 16 / 0 | +113 | +4 | 63 s |
| 14 | Row 4 + 4 GiB of experts in RAM (`--cpu-offload-gb 4 --cpu-offload-params experts`) | 5.63 | 39.25x | 16 / 0 | +112 | +4 | 259 s |

Row 14: model loading drops from 17.88 to 13.93 GiB, and vLLM's UVA offloader logs "Total CPU
offloaded parameters: 4.01" [measured]. UVA means the GPU reads the offloaded weights
from pinned RAM over PCIe as it runs. The likely cost is that traffic on every decode step
[inference; not profiled].

\* tm eager's v2 took 229 s because one item (trap2-mount) ran to the 4,096-token cap at
23.7 tok/s single-stream (4,096 / 23.7 ≈ 173 s) [measured; arithmetic]. On every graph server the
same item finishes in 56–86 tokens (references: 60) [measured].

Verdicts:

- **Adopt:** tm (row 2) and graphs at 0.92 (row 4). For headroom beyond 16 requests or 16k
  tokens, add fp16 state and `none` mode (row 7). `none` mode costs only prefix caching, and
  that hits 0–2.3% on these prompts because they are shorter than one 4,176-token block
  [measured earlier, gpu-lab KV RAM-tier test].
- **Refuted: b512.** A 512-token prefill batch, which wins on the dense Qwen3-14B pool,
  loses here: 1.61 vs 1.69 GiB of KV, a larger peak (1.22 vs 1.13 GiB), and it fails to
  start at 0.85 [measured].
- **Refuted for the eval: expert offload** (row 14). It has 3.3x row 4's KV, but the eval's
  16 requests already fit in row 4 with 0 waiting, so the extra room buys nothing and the
  slower decode costs 4x the wall time [measured]. Keep it for a context or batch that
  does not fit otherwise.
- **0.92 is vLLM 0.29.0's own default** (`gpu_memory_utilization: float =
  Field(default=0.92, …)` in `vllm/config/cache.py`). The repo's 0.85 is a step down from it,
  so adopting row 4 means deleting a flag [sourced].

## How the answers were checked

- **References:** each phase's per-item headline scores are compared with three identical
  G6q reference runs from the committed server (`results/research-eval-*-lightning-g6q-think-4k.json`
  and `results/noise/*-r2/-r3.json`). Their spread against each other sets the noise band.
  Pairs of reference runs agree on 156–157 of 158 v1 items, 130–133 of 134 v2 items, 5–7 of
  9 alert items and 12 of 12 trap3 items [measured].
- **Phases:** every phase agrees with the references on 156–158 v1, 126–132 v2 and 11–12
  trap3 items [measured, `results/kv-levers/agree.txt`, from `probes/kv_levers_agree.py`].
- **Changing the server flips a few more items than repeating it does.** v2 agreement is
  126–132 against 130–133 between references. That holds for plain graphs (row 4,
  128–132) too, which changes no storage format. Any change of kernels or batch shapes
  moves borderline greedy decodes, so a lever cannot be judged by a 1–4 item gap
  [measured; inference for the cause].
- **No lever moves the scores in one direction.** The signed sums fall on both sides of the
  references [measured]:

  | Split | Phases | References |
  |---|---|---|
  | v2 | +112 to +118 | 112–116 |
  | v1 | 127–129 | 128–129 |
  | trap3 | 4–6 | 5 |
  | alert | 3–6 | 3–7 |

  The extremes come from different levers: v2 +118 is the stack, v1 127 is b512 and noest,
  trap3 4 is plain graphs.
- **The state path does flip two v2 items, one each way.** Rows 5–7, 12 and 13 change the
  Mamba state (fp16, `none`, or both). Rows 2, 4, 8 and 14 keep it fp32 with `align`
  [measured]:

  | Item | Direction | State-changed servers (7) | State-kept servers (4) |
  |---|---|---|---|
  | `two_flag-chronyc` | fails; one extra tool call, 81 tokens vs 46 | 5 | 1 (row 14) |
  | `task-nft-13` | passes, where all references fail | 4 | 0 |

  Net zero on the signed sum. Both items are borderline, not broken: each flips in only
  some of the state-changed runs. `two_flag-chronyc` also flips on the offload server,
  which keeps the state but changes the expert path, so the state is the most common
  trigger, not the only one.
- **One flip comes from tm itself.** `trap2-strace` passes, where all references fail, on 7
  of 11 tm servers, including tm eager, which differs from the references only by tm
  [measured]. A plausible cause is that unwrapped experts use a different fused-MoE kernel
  [inference].
- **The check can see a missing adapter.** The base model agrees with G6q on only 111–121 v1,
  92–97 v2 and 8–9 trap3 items [measured]. A server that silently dropped the adapter under
  tm would show up here. Alert (9 items) cannot tell base from G6q, so it is reported but
  proves nothing.

## Speed

| Server | Peak aggregate decode | v1 / v2 / alert / trap3 wall |
|---|---|---|
| Committed (7 running) | 124 tok/s | 100–110 / 95–102 / 17 / 10–11 s |
| tm eager | 266 tok/s | 53–55 / 229* / 10 / 6 s |
| tm graphs (9 runs, rows 4–8, 12, 13) | 355–396 tok/s | 33–37 / 30–34 / 4–6 / 3–4 s |
| tm graphs, 4 GiB experts in RAM (row 14) | 116 tok/s | 133 / 126 / 18 / 11 s |

[measured, peaks are vLLM's 10-second "Avg generation throughput" lines; walls are the eval's
`elapsed_s`]

Graph runs have no single-stream samples: the evals finish without a lone tail.

## Untested levers, ranked

1. **Raise `--max-num-seqs` above 16** now that 16 requests use under half the pool. It only
   helps clients that send more than 16 at once (the eval sends 16), and its value is
   aggregate tok/s, untested here [inference]. Graph capture sizes grow with it.
2. **Get the display off the NVIDIA card.**
   - Today gnome-shell and Xwayland draw on the dGPU: about 215 MiB, plus every open window
     (Antigravity 679 MiB, Chrome 208 MiB) [measured, earlier].
   - vLLM charges all of it to KV silently [measured, earlier]. Free memory at startup
     varied between 22.14 and 22.8 GiB across runs.
   - Options:
     - **Headless window** (`systemctl isolate multi-user.target`, drive over SSH): ~0.2 GiB,
       reversible, no BIOS change [inference].
     - **BIOS Hybrid (Optimus) mode,** so the iGPU drives the panel: 0.2–1.2 GiB depending on
       open apps [arithmetic].
   - The panel is wired to the dGPU today (eDP-1 on card1) [measured, `/sys/class/drm`].
   - WARNING: the BIOS "iGPU only" mode is not Hybrid. It makes Linux report "NVRM: No
     NVIDIA GPU found" [sourced, https://bbs.archlinux.org/viewtopic.php?id=306247].
   - The main gain is repeatable KV sizes, more than the GiB.
3. **A cgroup cap around pinned-RAM jobs.** Use `systemd-run --scope -p MemoryMax=…`, or
   `docker --memory` for containers. It frees nothing, but a runaway like the 70B prefetch
   (53 GiB pinned) would then die alone, not take the session with it [inference].
4. **TurboQuant 4-bit KV for the dense and DeltaNet models,** not Lightning.
   - Formats `turboquant_4bit_nc` and `turboquant_k8v4` are in vLLM 0.29.0 [sourced, image
     `config/cache.py`]. Published perplexity cost: +2.7% and +1.2% [sourced, its config
     docstring].
   - Per token, fp8 → TQ4 is 163,840 → 85,760 B for the 70B [arithmetic]: 52k → 100k tokens
     in 8 GiB. Qwen3.8 goes 32,768 → 16,768 B; Qwen3-14B 81,920 → 42,880 B.
   - Not proven on sm_120 [inference].
   - `nvfp4` KV is gated to compute family 100 and rejects sm_120 [sourced, `flashinfer.py`
     `supports_kv_cache_dtype`].
5. **70B resident on one card at about 2.25 bits per weight** (QTIP/EXL3 trellis).
   - Size: 70.55e9 × 2.25 / 8 = 18.5 GiB, leaving about 4 GiB for KV with no PCIe in the
     loop [arithmetic].
   - QTIP reports Llama-2-70B at 2 bits with C4 perplexity 5.48 vs QuIP# 5.71 [sourced,
     https://arxiv.org/abs/2406.11235, numbers from a search snippet].
   - Cost: vLLM 0.29.0 has no trellis loader [sourced, image grep], so this means an engine
     swap (ExLlamaV3, not installed) plus a quantization run.
   - The ceiling is large: today's one-card 70B decodes at 1.07–1.1 tok/s per stream over
     UVA [measured, O70].
6. **Hot/cold expert bits:** NVFP4 for often-routed experts, 2-bit for cold ones kept in RAM.
   - This is HOBBIT's precision-on-miss idea [sourced, https://arxiv.org/abs/2411.01433].
   - It needs a routing histogram and a loader that splits vLLM's stacked expert tensor:
     code, not a flag [inference].
   - The plain version costs 4x the eval's wall time for 4 GiB offloaded (row 14). Keeping
     the hot experts on the card, and only the cold ones in RAM, is how offload could get
     cheap. This is the lever to build if a workload ever needs row 14's room [inference].
7. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for QLoRA training:** zero cost to
   try, gain unknown [inference].
8. **70B QLoRA with gradient accumulation inside the layer loop.**
   - Run every microbatch through layer i before loading layer i+1. That cuts PCIe weight
     traffic by the accumulation factor [inference].
   - The cost is pinned RAM for the checkpointed activations: 16 sequences of 1,024 tokens
     come to about 20 GiB [arithmetic].

## "Octization": new formulas, judged

David's word, read four ways: 8-bit codes; octree/hierarchical codebooks; 8-way product
quantization; 8x compression. Each was aimed at the term it would shrink.

| Idea | Target | Result | Verdict |
|---|---|---|---|
| fp16 Mamba state | state page (binding) | sim: worst layer 6.1e-3 at 4k steps; in vLLM 1.47x concurrency, answers unchanged ×3 | **Adopt for ≤4k traces** |
| bf16 state | state | sim: 5.3e-2 at 4k | Kill |
| int8 per-row state (+fp16 scale), stochastic rounding variant | state | sim: 1.4e-1 / 2.1e-1 | Kill |
| fp8 per-row state | state | sim: 1.0 | Kill |
| Mixed per-head int8/fp16 state | state | 1.77 B/element vs fp16's 2 (1.13x), needs a kernel | Kill: not worth a kernel |
| Expert = reference + delta | expert weights (15.39 GiB) | distance 1.28–1.42, \|cos\| ≤ 0.041; aligned 1.33–1.42 | **Dead** |
| Shared-exponent blocks (E8M0 per 64 + half-step bit per 16) | expert weights | 4.19 vs 4.5 bits per weight: −7%, ~1.1 GiB, new GEMM kernel | Kill |
| 8-way product quantization of KV (CommVQ-style) | attention KV | 192 B/token vs 3,072 fp8, but KV is ~1/3 of a page here and needs a trained codebook | Kill for Lightning |
| Hot/cold expert bits | expert offload traffic | untested | Defer (untested list) |

The state simulation (`probes/mamba_state_precision.py`, `results/mamba-state/`):

- **What it replays:** vLLM's decode recurrence with the checkpoint's real A, dt bias and D
  for all 23 Mamba layers × 64 heads. Inputs are synthetic Gaussian; the state is stored in
  each candidate format between steps, and each output y is compared with fp32.
- **Pass bar:** relative error of y under 1e-2.
- **fp16 at 4,096 steps:** worst layer 6.1e-3, worst head 9.2e-3, all 1,472 heads pass. The
  largest state value is 377, far below fp16's 65,504 [measured].
- **fp16 at 16,384 steps:** worst layer 1.5e-2, still growing. Only 38% of the slowest-decay
  heads (decay below 1e-4) pass [measured, `long16k-mu0.txt`]. That is the reason for the
  ≤4k limit.
- **Shorter state memory (dt shifted down, `mu-1.txt`):** fp16 still passes [measured].
- **Caveat:** a pass is necessary, not sufficient. Real activations have outliers that
  Gaussian inputs lack, which is why the vLLM eval rows above are the actual evidence.

The expert-delta test (`probes/expert_delta.py`, `results/expert-delta/out.txt`):

- **Method:** it reads Lightning BF16's routed experts for layers 1, 27 and 51 (128 each;
  up 1,856 × 2,688, down 2,688 × 1,856). For each expert it takes the distance to its
  nearest neighbour relative to its own norm. Delta coding would need under 0.5 to pay;
  above 0.9 it is dead.
- **Result:** 1.28–1.42.
- **Alignment does not rescue it:** the hidden neurons of each pair were matched greedily by
  best |cos|, a looser match than any real permutation. Median 1.33–1.37 (up) and 1.40–1.42
  (down).
- **The experts are close to mutually orthogonal:** pairwise |cos| ≤ 0.041, and √2 = 1.41 is
  the distance between two orthogonal matrices of equal norm.
- **Zero rows:** layer 1 has 1,372 all-zero up rows (0.6%) but no all-zero experts.

## OS and system

- **Distro swap: no VRAM effect** [inference]. What the card holds depends on the driver and
  the kernel module. Ubuntu 26.04 already runs the open 595 module, and CachyOS, Fedora,
  NixOS and Pop!_OS ship the same one.
- **`CUDA_MODULE_LOADING=LAZY`: already the default** since CUDA 12.3 [sourced,
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html].
- **GSP firmware heap: no knob.** The open module requires GSP [inference].
- **zram/zswap: cannot help pinned memory** (UVA offload, layer streaming), because pinned
  pages are never swapped [inference].
- **Filesystem compression: −21.6% on an NVFP4 shard, −21.8% on BF16** with zstd -3
  [measured]. The disk has 252 GiB free, though, and the root filesystem would need a
  reformat. Wrong problem.
- **Prompt compression (LLMLingua-2): no memory gain.** Prompts are 415–1,210 tokens, under
  one KV block, so it saves prefill time only, and it needs an install [measured; inference].
- **Thinking traces are tail-driven.** The base model's thinking-on v1 averages 251
  completion tokens (p90 571), and G6q's around 40. Truncation, not the mean, sets the
  wall time (the tm eager v2 item above) [measured, earlier eval JSONs].

## Training levers

Already in place:

- **Cut cross-entropy:** `g6_train.py` uses CCE, so logits are not materialized [measured,
  source].
- **`--no-fp32-upcast`:** the flag that makes 32B fit. PEFT upcasts the 151,936 × 5,120
  embedding to 2.90 GiB of fp32 [measured]. It should be the default for any model with a
  large vocabulary: Qwen3.8's embedding would be 4.74 GiB at fp32 [arithmetic].
- **Attention in bf16 plus the "lean scan"** for Lightning 4-bit LoRA [measured, docs].
- **NF4 layer streaming** for the 70B [measured].

Killed, because they shrink a term that is about 2% of the budget, or less, with LoRA r16:

- 8-bit or 4-bit optimizers, GaLore, Q-GaLore and APOLLO. An 8B LoRA's Adam state is about
  0.5 GiB [arithmetic].
- DoRA (more activation memory, not less), and LoRA-FA or VeRA, each under 1 GiB at r16
  [inference].
- BitDelta, which targets full fine-tunes, not LoRA [sourced].
- CUDA managed memory or HMM oversubscription: more than 3x slower once oversubscribed
  [sourced, https://arxiv.org/abs/1910.09598]. Explicit pinned copies (UVA, streaming)
  already do the same job faster.
- CPU expert compute (kTransformers-style). The 275HX has AVX2 and AVX-VNNI, no AVX-512, no
  AMX [measured, `/proc/cpuinfo`], while those systems' speed comes from AMX [sourced].

## Corrected figures

These earlier figures from this study's research notes are superseded:

- **Routed experts are 15.39 GiB** (5.35 MiB per expert), not 17.77 [measured, safetensors
  headers].
- **Mamba state is 46.8 MiB per request at fp32** (48.8 with the padding layer) and 23.8 MiB
  at fp16 [arithmetic].
- **"73 MiB per 5.1k request" is wrong** in align mode, which takes two blocks per Mamba
  group (the block model above).
- **"LoRA forces eager" is wrong.** LoRA needed memory that 0.85 did not leave; with tm at
  0.92, graphs run [measured].
- **Merging the adapter would gain at most 0.02 GiB beyond tm** (17.88 vs 17.86 GiB of
  weights), and it would mean a new checkpoint per adapter [measured; inference].
- **"Graphs make the eval ~3x faster" holds only against the committed server,** where it
  includes the end of queueing. Graphs alone are 1.4x on peak aggregate [measured].

## Adoption asks (David)

Nothing in `probes/` or `bin/lab` was changed. Each switch below changes a reference
condition, and these scripts produce the comparison runs, so changing them is David's call:

- **The G6q/G6u eval servers.** `probes/g6_eval.sh`, `noise_eval.sh`, `g6t_collect.sh`,
  `g7_lora_smoke.sh` and `g7b_desktop.sh` serve with `--enable-lora --enforce-eager
  --gpu-memory-utilization 0.85`; `s1_screen.sh` with `--enable-lora --max-num-seqs 16`.
  Proposal: add tm, drop `--enforce-eager`, and set 0.92 (rows 2 and 4). For more room,
  add `--mamba-ssm-cache-dtype float16 --mamba-cache-mode none --no-enable-prefix-caching`
  and `-e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` (row 13).
  - A run that is compared with existing results should keep the old flags, or rerun the
    references under the new ones. Every server change flips 1–4 items more than a repeat
    does (row 4 alone does), so mixing old and new servers in one comparison adds noise
    [measured]. New references cost three runs of each split. On one graph server that is about 6
    minutes: ~100 s to start, then ~75 s per run of all four splits [arithmetic from the
    walls above].
- **`bin/lab`'s Lightning serving** (gpu-lab repo): the same flags. At 0.92 vLLM leaves
  1.9 GiB of the card for everything else, and the desktop session and apps already take
  0.2–1.1 GiB of it. So close apps before starting, or get the display off the card
  [arithmetic; measured earlier for the apps].

## Reproduce

All on the laptop, with Chrome and Antigravity closed, from the worktree:

```
bash probes/kv_levers.sh round4    # rows 2-9 (one server per phase, evals into results/kv-levers/)
bash probes/kv_levers.sh round5    # rows 10-14
python3 probes/kv_levers_agree.py > results/kv-levers/agree.txt
```

The CPU-only probes run in the serving image (`--network none --pull never`). The command
lines are in each probe's docstring:

- `probes/mamba_state_precision.py` reads the NVFP4 snapshot.
- `probes/expert_delta.py` reads the BF16 snapshot `a9904d2`.

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
6. **Expert offload to RAM triples KV. It is the path for long-running agent contexts,
   and it is being optimized, not judged** (David, 2026-10-05).
   - `--cpu-offload-gb 4 --cpu-offload-params experts` moves 3.95 GiB of experts to RAM.
     KV goes from 1.69 to 5.63 GiB, 141 to 471 blocks. At 128k contexts that is 3.5 vs
     11.8 requests at once, and the longest single context goes from 555k tokens to 1.93M
     [measured: vLLM prints 3.52x and 11.78x; arithmetic for the rest].
   - Lightning's full 1,048,576-token context needs 3.11 GiB of KV. Without offload it
     does not start: vLLM refuses it against 1.68 GiB available and names 551,232 tokens
     as the longest it could take (p8-1m-base) [measured]. With offload it serves a needle
     at 1,032,631 tokens, 8 of 8: 954 s under UVA 4 GiB, 791 s under K6 + P1 (row G).
   - Its cost on the short eval: v1 + v2 take 259 s against 63–71 s on the card, and
     peak decode falls from 355–396 to 116 tok/s [measured]. The hot-expert cache with
     staged prefill (K6 + P1, rows K and U) takes 99 s, with the same scores.
   - That eval cannot judge it. Its 16 requests already fit without offload (row 4: 16
     running, 0 waiting), so it sees only the cost. The workload offload is for is a few
     agents whose contexts grow past what the card holds. There, a request that does not
     fit is a failure, not a slowdown. "Experts in RAM" below measures that workload and
     lists every option on the path.
   - Where it stands (2026-10-08, laptop). The hot-expert cache (K6, row K) with the live
     miss copy and LFU eviction decodes at 175–181 / 260 / 366–419 tok/s at 1 / 4 / 16
     streams over its rounds (p20–p32), against UVA's 76 / 98 / 132. A fill ahead on the
     copy engine (row O: K = 3, from 8 tokens) adds 5% at 16 streams (p32: 389 and 388
     against 366 and 372) and is neutral at one. Staging prefill on the copy engine (P1,
     row U) takes a 98k prefill from K6's 25.3 s to 18.0 s (no offload 15.4). Agents to
     127k (rows D and I): prefix KV in RAM alone, at 4,096-token chunks, runs 8 agents in
     351 s and 16 in 697 s (two draws each); K6 with a 2 GiB cold tier and KV in RAM, at
     8,192, ties it (352 s and 698 s, one draw) with 1.6× / 1.3× better late TTFT, and a
     3 GiB tier buys 3.0× / 2.8× for 2–3% of wall. No offload takes 2,461 s for eight (row B).
     Every arm that recomputes nothing computes the same 1.43M uncached tokens for eight
     agents and 2.86M for sixteen, at no more than 4,114 tok/s [arithmetic]: the memory
     levers have reached the prefill-compute floor. A faster MoE kernel does not lower it
     (row V): FlashInfer's B12x prefills 98k 21% faster, but at equal KV it gains 1–2% on
     agents, and its real KV is 0.75 GiB smaller, which costs 9–12%. Traces (p39) show
     where the rest goes: past 100k tokens of context, a 4,096-token prefill step is 60–64%
     attention and 29–32% MoE, with or without the cold tier. The deep end of the load is
     bound by attention over the prefix, not by where the experts live (row L). Swapping
     the attention kernel does not lower it (p40): Triton cuts late TTFT 11% but costs 17% of
     wall (417 s vs 357 s), and capturing every decode size to 8 changes nothing. Nor
     does bf16 KV (p41–p44): it decodes 28–60% faster in about half the tokens, and no
     bf16 arm beat its fp8 twin's wall. Two cards do: the pool holds 4.8M tokens and
     finishes sixteen agents in 615 s, 12% sooner than the laptop; the desktop's 3090
     alone takes 884 s.
     KV in RAM at 8,192 can deadlock vLLM 0.29.0's scheduler; `KSTAGE_UNJAM=1` clears it.
     The full 1M context serves with offload and not without (row G). The CPU computes a
     cold expert no faster than the copy engine moves one (row P).
7. **A different Linux would not free VRAM.** The one OS-level lever is getting the display
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
- **Expert offload (row 14): not judged by this eval.** It has 3.3x row 4's KV with the
  same answers. On this eval it only costs: the 16 requests already fit in row 4 with 0
  waiting, and the slower decode takes 4x the wall time [measured]. Its workload is long,
  growing agent contexts, measured in "Experts in RAM".
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

## Experts in RAM

David's direction (2026-10-05): offload is the path for long-running agent contexts. Keep
optimizing it, and do not close it until every known, experimental and potential option
on it has been tried. This section is that program. Its verdicts name a workload.

### What it buys: blocks, exactly

A request takes ceil(tokens / 4,176) attention blocks plus 8 state blocks (4 Mamba groups
× 2 in `align` mode), at 12.24 MiB a block ("Why the 16k concurrency moves in steps").
Row 4 has 141 blocks; row 14 has 471 [arithmetic; it reproduces vLLM's 3.52x and 11.78x
at 131,072 tokens].

| Workload | Blocks | Row 4 (141) | Row 14 (471) |
|---|---|---|---|
| 4 agents growing to 98k | 128 | fits | fits |
| 4 agents growing to 127k | 156 | no | fits |
| 6 agents growing to 98k | 192 | no | fits |
| One 524,288-token context | 134 | fits | fits |
| One 1,048,576-token context | 260 | no | fits |

vLLM's "GPU KV cache size: N tokens" line is concurrency × `--max-model-len`, not a fixed
capacity. The same 1.69 GiB prints 192,512 tokens on a 16k server and 462,028 on a 128k
server, because the 8 state blocks are per request [measured, p6-base]. Compare blocks or
GiB, never that line across servers.

### The workloads

`probes/offload_bench.py` runs them against a live server; `kv_levers.sh` phases with
`WORK=` start the server and save `bench-TAG.json`:

- **decode:** C parallel streams, 1k-token prompts, 256 tokens each: per-stream and
  aggregate tok/s.
- **prefill:** one prompt per length, cold and then warm (prefix-cache hit).
- **agent:** N agents in parallel. Each turn adds 4,096 tokens of tool output plus 256 of
  reply, until the target length. Records per-turn time to first token, cached tokens,
  preemptions and wall time. 4 × 98k fits both rows, so it measures the cost; 4 × 127k
  and 6 × 98k do not fit row 4, so they measure what the room buys.
- **needle:** eight "the access code for NAME is NUMBER" lines in real text, scored by
  exact match, at lengths up to the full context.
- **experts:** which experts each layer routes to, on the eval prompts and agent text
  (`--enable-return-routed-experts`).

### The options and where each stands

Laptop, 131,072-token server, unless a row says otherwise. Decode is aggregate tok/s at
1 / 4 / 16 streams; "98k" is the cold prefill of a 98,304-token prompt.

| ID | Option | How | Result | Status |
|---|---|---|---|---|
| A | UVA offload vs none | `--cpu-offload-gb 4 --cpu-offload-params experts` | KV 1.69 → 5.63 GiB (3.3x); decode 202 / 430 / 701 → 76 / 98 / 132; 98k 15.4 → 34.8 s | done |
| B | Agent loads that overflow row 4 | 6 × 98k, 4 × 127k | 6 × 98k: no offload 730 s, 10 preemptions, late TTFT 19.2 s; off4 572 s, 0, 3.8 s; K2 gather4 402 s. 4 × 127k: 519 vs 520 s, but 195 s with no expert offload and evicted prefix KV kept in RAM (row I). 8 × 127k (3.74 GiB running, p13): no offload 2,461 s, 80 preemptions, late TTFT 68.4 s; prefix KV in RAM (8 GiB) 374 s, 0, 9.2 s; K6 589 s, 0, 3.5 s (decode median 14.3 tok/s a stream vs 33.1); K6 + P1 (p15) 447 s, 0, 2.49 s, 18.5 tok/s | 4 × 127k is prefix eviction, not running room. At 8 × 127k host KV wins wall time and K6 wins time to first token: K6 holds all 8 contexts on the card, but its prefill chunks are 1.6x slower (row K). P1 (row U) takes K6 from 589 to 447 s and its late TTFT to 2.49 s; host KV still finishes 16% sooner, with a 3.7x longer late TTFT [arithmetic]. 16 × 127k (7.49 GiB, past K6's 5.71; p14, 448 turns, 0 errors): prefix KV in RAM (16 GiB) 744 s, decode median 28.3 tok/s a stream, late TTFT 25.0 s; K6 alone 5,816 s, 6.6 tok/s, 263 s; K6 + prefix KV in RAM 1,177 s, 7.2 tok/s, 7.1 s, 12 preemptions. K6 alone found 9.73M of its 29.99M prompt tokens in the prefix cache and recomputed the other ~20M at its slow prefill [arithmetic]: that is what P1 (row U) attacks. K6 + P1 (one staging buffer, p16, same `KSTAGE_COPY=82` as p14): 3,766 s, 9.3 tok/s, late TTFT 159.5 s, 0 preemptions; with prefix KV in RAM too 850 s, 9.8 tok/s, 4.74 s, 13 preemptions [measured, one draw]. P1 cuts K6 alone's wall time 35% and K6 + prefix KV's 28%, but host KV alone (744 s) still finishes first: the best K6 arm takes 14% longer [arithmetic]. With the live miss copy (row K, p22, 2026-10-07, one draw each): 8 × 127k K6 + P1 + `KSTAGE_COPY_LIVE=82` 434 s, 19.2 tok/s, late TTFT 2.47 s; with LFU too 418 s, 19.9 tok/s, 2.47 s; 16 × 127k, that + prefix KV in RAM 817 s, 10.2 tok/s, 4.70 s, 13 preemptions [measured]. The best K6 arms now take 12% / 10% longer than host KV alone (374 / 744 s), with a 3.7x / 5.3x shorter late TTFT [arithmetic]. Fewer experts in RAM (p23, 8 × 127k, live + LFU, one draw each): `KSTAGE_COLD_GB=3` 389 s, 21.8 tok/s, late TTFT 2.37 s, 1.74 misses a layer-step (4 GiB: 2.80), 4.50 GiB KV (1.23M tokens); 2.5 GiB 395 s, 21.8 tok/s, 2.39 s, 1.27 misses, 4.01 GiB [measured]. 3 GiB closes 29 of the 44 s gap: K6 now takes 4% longer than host KV alone, with a 3.9x shorter late TTFT [arithmetic]. 2.5 GiB cuts misses another 27% and gains nothing, so on this load misses stop being the bound at 3 GiB [inference]; the cold share is a per-load setting, sized to the KV the load runs |
| C | Offload size, 2–16 GiB | UVA | KV 3.66 / 5.63 / 9.57 / 13.49 / 16.83 GiB; c=16 231 / 132 / 77 / 54 / 42 | done |
| D | Larger prefill batches | `--max-num-batched-tokens 8192 / 16384` | 8192: 98k 34.8 → 27.2 s, c=16 132 → 141, KV 4.82 GiB; 12,288: 98k 26.4 s, c=16 143, KV 4.84 GiB; 16384 OOMs at startup. Agent loads (p31, p33–p36; one draw unless two are given), K6 + live copy + LFU, 8 × 127k: 2,048 418 s, decode 20.0, late TTFT 2.48 s, KV 5.44 GiB; 4,096 386 s, 22.8, 2.88 s, KV 5.55; 8,192 381 and 382 s, 22.3, 2.48 s, KV 4.63 (peak activation 1.14 → 1.32 GiB); 8,192 with ahead K3-min8 377 s, 22.4, 2.48 s, KV 4.34; 12,288 378 s, 25.8, 4.13 s, KV 4.65. With 3 GiB cold (`KSTAGE_COLD_GB=3`, p35; KV 3.69 GiB, 1,009,254 tokens), 8,192: 768 s, 23.5, 2.72 s, since 8 × 126,976 = 1,015,808 tokens no longer fit [arithmetic] and the 2.77M tokens that host KV brings back from RAM in the next arm are recomputed [inference]; with host KV 359 s, 24.9, 2.71 s. 16 × 127k, K6 + host KV (row I): 2,048 817 s, 10.2, 4.70 s, 13 preemptions (p22, p29); 4,096 777 s, 11.2, 5.71 s, 34; 8,192 765 and 769 s, 12.1, 8.02 and 8.12 s, 18 and 24, KV 4.66; 3 GiB cold, 8,192 716 s, 14.8, 12.93 s, 19. Host KV without K6, 8 agents / 16: 2,048 (KV 1.69 GiB) 369 s, late TTFT 9.10 s / 742 s, 25.0 s; 4,096 (KV 1.80 GiB, 491,520 tokens) 351 and 351 s, decode 36.2 and 36.3, 8.24 and 8.27 s / 696 and 697 s, 22.5 and 22.6 s, 0 preemptions; 6,144 unjammed (p36; KV 1.19 GiB, 324,403 tokens) 378 s, 47.3, 12.6 s, 28 preemptions / 769 s, 29.6 s, 103, with one stall at 16 that unjam cleared in 2.0 s; 8,192 (KV 0.88 GiB, 239,206 tokens) deadlocks at both (Running 0, 0 tok/s, for hours); unjammed (`KSTAGE_UNJAM=1`, below) it finishes: 415 s, 15.3 s, 40 preemptions / 835 s, 33.5 s, 169. K6 with host KV across cold tiers (p37; GPU KV 1.79 / 2.71 / 3.69 / 4.61 GiB at 1 / 2 / 3 GiB cold at 8,192 and 3 GiB at 4,096): 2 GiB at 8,192 352 s, 29.2, 5.05 s, 5 preemptions / 698 s, 19.0, 17.8 s, 27; 3 GiB at 8,192 again 359 s, 2.72 s, 0; 3 GiB at 4,096 363 s, 2.85 s, 0 / 716 s, 13.3, 8.02 s, 10; 1 GiB unjammed 354 s, 34.6, 8.30 s, 21. Every arm that recomputes nothing computes the same uncached tokens (queries less prefix and host-KV hits): 1,431,808 for 8 agents, 2,863,616 for 16, at 3,944–4,114 tok/s; 3 GiB without host KV recomputed 4,204,672 (p35, 768 s) [measured] | done: for K6, 8,192 cuts wall 9% at 8 agents and 6% at 16; 4,096 matches it at 8 (386 s) and trades 12 s of wall for the best 16-agent late TTFT (5.71 s). 12,288 adds 1% and costs TTFT. For host KV alone 4,096 is the best measured (two draws each). With host KV, K6 at 2 GiB cold and 8,192 ties host KV alone (352 s / 698 s, one draw) with 1.6× / 1.3× better late TTFT (5.05 s against 8.27; 17.8 against 22.6); 3 GiB trades 2–3% of wall for 3.0× / 2.8× (2.72 s at 8,192; 8.02 s at 4,096). The wall is now prefill compute [arithmetic]: the same uncached tokens at ≤ 4,114 tok/s in every arm, so no memory lever can cut it further. 4 GiB cold at 4,096 keeps the best 16-agent late TTFT (5.71 s). 6,144 and 8,192 starve host KV alone of KV even unjammed |
| E | Prefetch whole layers | `--offload-backend prefetch`, G 4/2, step 1–2 | g4s2: KV 4.93 GiB, 98k 17.1 s (no offload: 15.4), but decode 9.9 / 38.7 / 149; pins ~2.5x the offload in RAM. g4s2 and g2s2 computed wrong outputs: vLLM's slot reuse races when the offloaded count is not a multiple of the step (below); their speeds stand. Step 1 (safe), 6 × 98k: 660 s. With the fix (`KSTAGE_PFFIX=1`), g4s2: KV 4.93 GiB, decode 9.9 / 38.4 / 148, 98k 17.6 s, host +12.1 GiB; 6 × 98k 650 s, late TTFT 2.0 s. Race-free G4/K2 step 2: KV 8.19 GiB, decode 5.7 / 22.8 / 88, 98k 18.7 s, host +17.8 GiB | done: the best 98k prefill of any offload arm except K2 with DMA at batch 8,192 (15.1 s, row K), at a tenth of the decode speed. Agent wall loses to K6 (650 vs 282 s) |
| F | Offload + the state levers | fp16 state, `none` mode | UVA 4 GiB + fp16 state: KV 6.28 GiB, decode 77 / – / 136, 4 × 127k 511 s (UVA alone 520). Adding `none`: 15.62x printed, but no prefix cache, so 4 × 127k took 2,474 s, late TTFT 33.9 s | done: fp16 state is free; `none` is unusable for agents (each turn recomputes its whole context). ESTCG=0 not run with offload |
| G | Full context: 512k, 1M | needle | round 8 invalid: the model answered with the file's own NAME = NUMBER constants, and 1M overran `max_model_len` (HTTP 400). 512k (round 10, no expert offload): 8 of 8 at 32k, 131k, 264k and 509,885 tokens, the last in 214 s; fp16 state the same. 1M (UVA 4 and 2 GiB): the bench never sent a prompt. Its rescale stopped at 1,198,435 tokens after 4 tries, because the text's characters per token vary along it; it now brackets the target and interpolates. p15, needle at 1,032,631 tokens: UVA 4 GiB 8 of 8 in 954 s (KV 5.63 GiB, 1.81 requests at once); K6 + P1 8 of 8 in 791 s (5.17 GiB, 1.66). Without expert offload vLLM refuses 1M (needs 3.11 GiB of KV, 1.68 available; p8-1m-base) | done: 1M serves only with offload; K6 + P1 is 17% faster than UVA |
| H | Eager under offload | `--enforce-eager` | UVA 4 GiB: KV 5.63 → 7.03 GiB (peak activation 1.13 → 0.18, weights + non-torch 14.76 → 14.31); decode 76 / 98 / 132 → 24 / 85 / 133. One stream runs at a third, 16 streams unchanged | done: KV for many streams, not one |
| I | KV blocks evicted to RAM | `--kv-offloading-size 16 --kv-offloading-backend native` | no expert offload, 4 × 127k: 195 s (no offload 519, UVA 4 GiB 520), late TTFT 2.50 s, 0 preemptions. 2.20M of the 7.50M prompt tokens came back from RAM; recomputing them would take ~300 s [arithmetic]. Host RAM +20.9 GB | done: the fix for prefix eviction, at no VRAM cost. It adds no running room. With K6, 8 × 127k: 579 s vs K6 alone 589 s, 0 external hits: K6's 5.72 GiB holds all eight (312 of 478 blocks) [arithmetic], so nothing is evicted. 16 × 127k (p14): 744 s, late TTFT 25.0 s, 27.1M tokens back from RAM, host +20.6 GiB; with K6, 1,177 s, 7.1 s, 18.0M from RAM, host +36.6 GiB (22.4 GiB still available). At 8,192-token chunks host KV alone deadlocks vLLM's scheduler (below; `KSTAGE_UNJAM=1` clears it); with K6 it ran 16 × 127k in 765 s (p31, row D). Alone, its best chunk is 4,096: 351 s at 8 agents, 696 s at 16 (p33; p35 repeats them within 1 s); 6,144 loses, 378 s / 769 s (p36). With K6 at 3 GiB cold: 359 s / 716 s (p35, row D); at 2 GiB, 352 s / 698 s (p37) |
| J | Routing histogram | `--enable-return-routed-experts` | the median layer's 32 most-routed experts of 128 take 0.52 (code) / 0.63 (eval) of routings; uniform would be 0.25 | done; drives K |
| K | Hot experts on the card, cold in RAM | `probes/kstage` plugin: K2, K5, K6 | below. Laptop K6: decode 166 / 241 / 294 at 5.71 GiB KV (UVA 76 / 98 / 132 at 5.63), 6 × 98k in 282 s (UVA 572, K2 402); with P1, the live miss copy and LFU (p20) 176 / 260 / 419 at 5.44 GiB | K2, K5, K6 measured on both machines; K6's prefill (98k 25.3 s vs 15.4 none) was the open cost; P1 (row U) takes it to 18.0 s, 13.6 s at batch 8,192 |
| L | Where offload's time goes | copy probes; torch profiler (p39) | read rates: Marlin through UVA 27.2, Triton gather 36.8, copy engine 50.9 GB/s (desktop 7.2 / 11–12.5 / 12.2). 8 agents past 100k: a 4k prefill step is 60–64% attention, 29–32% MoE (p39) | traced; p40: Triton attention and full decode capture do not lower the wall; p41–p44: neither does bf16 KV |
| M | Offload on the training side | QLoRA | | open |
| N | Expert caches outside this vLLM | vLLM PR #37190 (`probes/pr37190.sh`), LMCache (`-lmcN` arms of `probes/desktop_agents.sh`), llama.cpp (`probes/llamacpp/`), ktransformers | **PR #37190** (vLLM 0.29.0 with the PR applied, `probes/pr37190/`; `results/pr37190/`, laptop, 2026-10-08) caches BF16 and FP8 experts only, so it served Lightning's BF16 checkpoint quantized to FP8 at load, on the Triton MoE kernel, without G6q (the PR refuses LoRA): 64 of each layer's 128 experts on the card (`--moe-expert-cache-size 64`), the rest pinned in RAM. An FP8 expert is 9.5 MiB, 27.4 GiB for all 2,944, against NVFP4's 5.35 MiB [arithmetic]. PyTorch rounded each layer's pinned block up to a power of two (host +48.8 GiB) until `pinned_max_round_threshold_mb:64` (+31.0 GiB). KV 3.14 GiB, 861,798 tokens. Decode 33.2 / 104.8 tok/s aggregate at 1 / 16 streams; 98k 33.08 s (8k 2.38, 32k 10.26). 8 × 127k with 8 GiB of host KV: 1,012 s, decode median 8.4 tok/s a stream, late TTFT 5.81 s, 0 preemptions, host +38.7 GiB, against p35's host KV alone 351 s, 36.3, 8.27 s (row I) [measured, one draw]. It computed 1,431,808 uncached tokens, the same as every arm in row D that recomputes nothing [arithmetic], so the 2.9x is time per token, not recomputation. Splitting an overfull batch by tokens instead of experts (`--moe-expert-cache-split token`, p5): 8k prefill 38.51 s cold, 32k 160.4 s, 98k 408.9 s (12.4x the default split); decode 34.1 / 51.1 tok/s aggregate at 1 / 16 streams [measured]. **LMCache** LMCache 0.5.4 (the one in the v0.29.0 image) caches KV, not experts; it is here as the other host-memory tier the agents' KV could use, on the desktop (`results/desktop-agents/`, 2026-10-08). Lightning is hybrid, so it runs as LMCache's multiprocess server beside vLLM, through `LMCacheMPConnector`, with three patches for vLLM 0.29.0 (`probes/lmcache/v029_layout.py`), 4,176-token chunks (one vLLM block; the connector takes only multiples) and 8k batches. Against d41, native host KV with 8 GiB at 8k batches (443 s, decode median 28.5 tok/s, late TTFT 9.19 s, 1,431,808 tokens computed), 8 agents: 8 GB with LMCache's default eviction, which drops 20% each time it passes 80% full (d48), 1,737 s, late TTFT 42.04 s, 6,894,016 tokens computed (4.8x); 8 GB evicting 5% at 95% (d49) 1,162 s, 21.58 s, 3,515,632 (2.46x); 12 GB (d51) 935 s, 14.72 s, 2,408,992; 16 GB (d52) 929 s, 14.69 s, 2,400,640 (1.68x). Decode 23.9-25.3 tok/s, preemptions 11-20 against 8; host RAM +12.5 / +16.6 / +19.5 GB at 8 / 12 / 16 GB [measured, one draw each; tokens computed = queried - prefix hits - external hits, arithmetic]. From 12 GB on, memory changes nothing: external hits 10,632,096 at 12 GB, 10,640,448 at 16 GB [measured]. The 968,832 tokens d52 computes beyond d41 are 4,325 a turn, 1.04 chunks [arithmetic], which fits LMCache storing and returning whole 4,176-token chunks, so each turn recomputes about one chunk native host KV keeps [inference]. Recompute is not the whole gap: 1.68x the tokens took 2.1x the wall, with decode 13% slower [arithmetic]. LMCache and vLLM both send usage reports by default (stats.lmcache.ai, stats.vllm.ai); d48-d50 ran with them on, d51 and d52 with `DO_NOT_TRACK=1` (Adoption asks). **llama.cpp** (master de7fa0a, 2026-10-08, built in the v0.29.0 image for sm_86 and sm_120a by `probes/llamacpp/build.sh`; arms in `probes/llamacpp/serve.sh`, logs in `results/llamacpp/`, the desktop's suffixed `-3090`) serves a GGUF that its own `convert_hf_to_gguf.py` made from the NVFP4 checkpoint (bee7596, `--no-mtp`; 19.8 GB, experts still NVFP4; Reproduce). It is not like for like with vLLM's arms: no LoRA (they served G6q), q8_0 or q4_0 KV against fp8, all of KV on the card in one 131,072-token slot per agent (q8_0: 6,528 MiB at 16, where vLLM pages to host KV), and Mamba state resumed from checkpoints llama-server saves near each prompt's end. The placements are llama.cpp's own flags: every expert on the card (`-ngl 999`), every expert in RAM (`--cpu-moe`), the experts of blocks 0 to N-1 in RAM (`--n-cpu-moe N`, which counts all 52 blocks, 23 of them MoE), and `--moe-cache-mib M`, a cache on the card of the experts kept in RAM (`src/llama-moe-cache.cpp`), whose only setting is its size. Ubatches of up to 32 tokens (a constant, `max_batch`) look their experts up in it, upload the misses and evict the least recently used; larger ones only read it, copying the experts it holds card to card and the rest from RAM, so prefill does not evict what decode uses [sourced]. Its log counts uploads for ubatches of up to 32 tokens only, so the link times below are lower bounds [sourced]. Without the cache, experts kept in RAM run on the CPU for ubatches under 32 tokens and are copied to the card for larger ones (`GGML_OP_OFFLOAD_MIN_BATCH`, `ggml/src/ggml-cuda/ggml-cuda.cu`) [sourced]. Laptop, 8 agents, against p35's host KV alone (351 s, decode median 36.3 tok/s, late TTFT 8.27 s): every expert on the card with q8_0 KV, at 512-token ubatches since 2,048 does not fit, 520 s, 15.6 tok/s, 2.47 s; with q4_0 KV (half of q8_0's bytes; not comparable to fp8) 2,048 fits (22,604 MiB on the card), 485 s, 16.4, 2.23 s (1,024: 507 s), llama.cpp's fastest arm and 1.38x vLLM; every expert in RAM without the cache 1,654 s, 4.7, 4.79 s; a 12,000 MiB cache (76% of the experts) with q8_0 KV at 4,096-token ubatches 603 and 604 s, 14.0 and 14.1, 3.36 and 3.34 s (two draws), hit rate 89.49% on ubatches of 1-8 tokens, 3,575 GiB uploaded, 74 s of the link at 51.8 GB/s; a 15,000 MiB cache (95%) with q4_0 KV at 2,048, 535 s, 15.2, 2.55 s, hit rate 99.24% / 91.42% on ubatches of 1-8 / 9-32 tokens, 257 GiB, 5 s, 1.10x every expert on the card [measured, one draw unless stated; link time = uploaded / 51.8 GB/s, arithmetic]. 16 agents, against 697 s: every expert on the card does not fit even with q4_0 KV, since the compute buffer (2,221 MiB at 512-token ubatches, 2,135 at 256) fails to allocate. With q8_0 KV the card holds an 8,500 MiB cache at 4,096-token ubatches (1,605 s, decode 4.9, late TTFT 4.62 s, hit rate 60.44% on ubatches of 9-32 tokens, 23,966 GiB, 497 s of link) or 10,000 MiB at 2,048 (1,578 s, 4.8, 3.35 s, 63.41%, 20,701 GiB, 429 s); q4_0 KV (3,456 MiB against 6,528) moves 10,000 MiB's wall 2% (1,553 s, 4.9, 3.15 s, 63.42%) and makes room for 13,000 MiB, 83% of the experts (1,288 and 1,270 s, 5.8 and 6.0, 2.75 and 3.00 s, 88.33% and 88.22%, 6,652 and 6,721 GiB, 138 and 139 s; two draws) [measured], so the hit rate, not the KV format, moves the cache's wall [inference]. The static split, with q4_0 KV at 2,048-token ubatches: `--n-cpu-moe 10` (4 MoE layers, 2,740 MiB, in RAM; 23,366 MiB on the card) 1,299 and 1,289 s at 8 threads (5.9 and 6.0, 2.83 and 2.78 s; two draws), tying the 13,000 MiB cache at the same card memory. Mixing the two, 7 MoE layers in RAM with a 2,000 MiB cache of them, 1,322 s (5.8, 2.69 s) at 8 threads, uploads 16,848 GiB (hit rate 41.62% / 2.73%), at least 349 s of link, and lands within 2-4% of both [measured], so on the laptop the link does not set the wall [inference]. More threads move the static split more than the cache: `--n-cpu-moe 10` ran 1,190 s (6.5, 2.74 s) at 16 threads and 1,177 s (6.6, 2.82 s) at 24, all of the 275HX's cores, 8-9% under its 8-thread draws; the 13,000 MiB cache ran 1,252 s at 16 (6.1, 2.73 s; 90.11% / 88.27%, 6,693 GiB, 139 s), 1-3% under its own. One more MoE layer on the card (`--n-cpu-moe 8`) does not fit at 2,048, nor do 4,096-token ubatches, since the compute buffer is 2,748 / 3,106 / 3,821 MiB at 1,024 / 2,048 / 4,096; at 1,024 it fits (23,720 MiB) and is the fastest static arm, 1,160 s (6.8, 2.87 s) at 16 threads and 1,163 s (6.8, 2.98 s) at 24; at 1,536 it loads (23,870 MiB) but CUDA's memory pool cannot grow (`cuMemCreate`, out of memory) on the first request; and `--n-cpu-moe 6`, one MoE layer fewer in RAM, does not fit even at 512 (compute buffer 2,569 MiB). With fewer experts on the card the cache wins: 10 MoE layers in RAM (`--n-cpu-moe 24`, 8,907 MiB of experts on the card) with q8_0 KV, 1,934 s (4.0, 3.62 s), 1.20x the 8,500 MiB cache and 1.23x the 10,000 MiB one [measured]. Copying the experts kept in RAM to the card for every ubatch, so that decode no longer runs them on the CPU (`GGML_OP_OFFLOAD_MIN_BATCH=1`), shortens the static split's wall a little: `--n-cpu-moe 10` at 16 threads 1,160 s (6.6, 2.70 s) against 1,190, and `--n-cpu-moe 8` at 1,024 1,145 and 1,146 s (6.8 and 6.8, 2.75 and 2.73 s; two draws) against 1,160, llama.cpp's fastest 16-agent arm on the laptop, 1.64x vLLM. A decode-only bench (512 tokens) shows the copies happen: one stream 117.8 tok/s against 73.7 (+60%), 16 streams together 258.0 against 251.5 (+2.6%), the card's PCIe receive rate 17,635 MB/s against 61 (median of `nvidia-smi dmon -s t`'s one-second samples over the run, 50 and 53) [measured]. Every expert on the card at 16 agents fits only with fewer slots than agents: llama-server then saves an idle slot's prompt, with its Mamba checkpoints, to a prompt cache in RAM (`--cache-ram`) and loads the best match into the slot an agent's next request gets [sourced]. With q4_0 KV, 2,048-token ubatches and a 16,384 MiB cache, 8 slots ran 1,047 s (15.1, 24.66 s; 22,604 MiB) and 10 slots 1,032 s (12.1, 20.70 s; 23,441 MiB); 12 slots do not fit (context creation fails at load). The cache loaded a saved prompt for 432 of 448 turns in both (all but each agent's first), peaked at 3,720 and 2,825 MiB, and saving and loading took 77.6 and 77.5 s of the run (median 177 and 179 ms a request) [measured]. The late TTFT is agents waiting for a slot, not prefill [inference]. So with no expert in RAM llama.cpp runs 16 agents 1.48x slower than vLLM, and its fastest arm with experts in RAM (1,145 s) costs 1.11x on top of that [arithmetic]: on the laptop most of llama.cpp's gap is the engine, not keeping experts in RAM [inference]. Desktop 3090, against d41 (443 s) and d44 (880 s): a 12,000 MiB cache, 8 agents, 1,061 s, decode 7.8, late TTFT 5.20 s, hit rate 89.55%, 3,557 GiB, 313 s of the link at 12.2 GB/s; 16 agents, the laptop's 10,000 MiB arm at 2,048-token ubatches with q8_0 KV, 3,810 s (the laptop's 1,578 s), decode 2.0, late TTFT 6.75 s, hit rate 63.47%, 20,657 GiB, 1,818 s of the link [measured; link time arithmetic]. With q4_0 KV at 2,048-token ubatches, 8 agents: every expert on the card 644 s (12.8, 3.11 s; 22,364 MiB), 1.45x d41; a 15,000 MiB cache 739 s (11.3, 3.74 s; 99.24% / 91.38%, 257 GiB, 23 s), 1.15x the card. 16 agents: the 13,000 MiB cache 2,306 s (3.3, 5.08 s; 89.97% / 88.26%, 6,700 GiB, 590 s); `--n-cpu-moe 10` 1,983 s (3.9, 4.78 s) at 8 threads, 1.16x faster at the same card memory; `--n-cpu-moe 8` (3 MoE layers in RAM, 23,806 MiB on the card) 1,817 s (4.2, 4.43 s) at 8 threads, 1,811 s (4.2, 4.38 s) at 10, the i9-10850K's cores, and 1,898 s (4.1, 4.45 s) at 4 [measured, one draw each]. Larger ubatches pay on the 3090 up to what fits: `--n-cpu-moe 8` at 2,560 tokens and 10 threads 1,712 s (4.5, 4.00 s; 23,986 MiB), llama.cpp's fastest 16-agent arm on the 3090, 1.95x d44, and 1,812 s (4.3, 4.42 s) at 2,048 and all 20 hardware threads; 2,816 loads (24,074 MiB) but CUDA's memory pool cannot grow on the first request, and 3,072 does not fit (compute buffer 3,463 MiB); `--n-cpu-moe 10` at 4,096 and 8 threads 1,888 s (4.1, 5.97 s; 23,840 MiB), 1.05x faster than at 2,048. Copying the experts kept in RAM for every ubatch (`GGML_OP_OFFLOAD_MIN_BATCH=1`), which helped on the laptop, loses on the 3090: 2,092 s (3.7, 4.13 s) at 2,560 and 10 threads, 1.22x slower [measured]. Copying 3 MoE layers of experts (2,055 MiB) takes 177 ms at 12.2 GB/s against 42 ms at the laptop's 51.8 [arithmetic], so the slower link costs more than the CPU's decode saves [inference]. Every expert on the card with 8 slots and the 16,384 MiB cache, as on the laptop: 1,369 s (11.7, 31.20 s; 22,364 MiB), the cache loading a saved prompt for 432 of 448 turns, peaking at 3,720 MiB, 75.2 s saving and loading [measured]; 1.56x d44, and the fastest arm with experts in RAM costs 1.25x on top of it [arithmetic]. **ktransformers** (source 2ac8cf9, 2026-10-08) cannot run Lightning's experts: kt-kernel computes silu(gate) × up in every CPU backend (AVX2: `operators/avx2/avx2_bf16_utils.hpp`), and Lightning's experts are relu² with up and down projections only (`mlp_hidden_act: relu2`, no `gate_proj` in the weight index) [measured]. It does not need AMX: an AVX2 path takes BF16, FP8 and GPTQ-INT4 experts (v0.5.3), not NVFP4, served through its own SGLang fork (`sglang-kt`) [sourced] | PR #37190: loses, 2.9x host KV alone's wall at 8 agents; 16 not run. Whether the cache or FP8 on Triton costs the time is unknown: FP8 without the cache does not fit (27.4 GiB of experts) [arithmetic]. Its token split loses 16x on an 8k prefill and is not worth an agent arm. LMCache: loses, 2.1x native host KV's wall at 8 agents on the desktop with 12-16 GB against its 8 GiB. Past 12 GB the gap is chunk granularity [inference], and the chunk cannot be smaller than vLLM's block, which Lightning's Mamba page sets; 16 agents not run, since it loses at 8. llama.cpp: loses on both cards, and most of the gap is the engine, not experts in RAM: with every expert on the card its wall is 1.38x vLLM's on the laptop and 1.45x on the 3090 at 8 agents, and 1.48x / 1.56x at 16 (10 / 8 slots and its RAM prompt cache). Its fastest 16-agent arms with experts in RAM, 1,145 s on the laptop (1.64x) and 1,712 s on the 3090 (1.95x), cost 1.11x / 1.25x on top of its own all-on-card arms. On the laptop the static split and the expert cache tie at equal card memory; on the 3090 the static split beats the cache by 1.16x, ubatches pay up to 2,560 tokens, and copying every ubatch's experts loses 1.22x. Not adopted. ktransformers: needs a relu² expert kernel in kt-kernel and NemotronH wiring in `sglang-kt` before anything can be measured; row P bounds the payoff, since CPU experts at NVFP4's 5.35 MiB already ran no faster than the copy engine, and BF16 / FP8 experts are 3.6x / 1.8x those bytes [arithmetic] |
| O | Fill K6 slots a layer ahead | `KSTAGE=predict` runs the next layers' gates on this layer's input | desktop (k7), one layer ahead: of the 2.63 cold experts a decode layer-step routes to, each token's top 6 predicted name 68% (fetching 2.68) and its top 10 83% (fetching 4.35); prefill 88% / 96%. Two layers ahead: 58% / 72% | measured (k11, desktop): a copy kernel beside a matmul takes the sum of both, not the longer (16 misses: 5.09 ms vs 1.95 + 3.14; 1 miss: 1.99 vs 1.97 + 0.06), so K6's in-kernel miss copies never hide behind compute. A fill ahead has to use the copy engine; P1 (row U) does that for prefill. Laptop, `probes/kstage/ce_probe.py` (2026-10-06): a CUDA graph can drive it. Its kernels write a copy plan into pinned memory and raise a flag (captured `cuStreamWriteValue32`); a host thread (`ce_helper.c`) issues the copies on its own stream and sets a flag the graph waits on (`cuStreamWaitValue32`). 23 layers each reading 160 MB on the card (~200 µs, decode-like), K expert rows (5.35 MiB) chosen on the device a layer ahead: per layer the handshake alone (K=0) adds 15–17 µs, K=1 52–60, K=2 121–124, K=3 258–264, with 0 bad of 22 / 44 / 66 rows checked (two runs, at most 2 or 1 steps queued) [measured]. A row alone takes ~108 µs at 51.8 GB/s, so K=1 hides ~60% of its copy, K=2 ~50%, K=3 ~25% [arithmetic]; in-kernel copies hide 0% (k11). Three rows need ~324 µs, more than the 200 µs of work, and the copies probably slow while the work saturates the card's memory [inference]. With no cap on queued steps (20 launched back to back) it hangs, three runs of three: the helper sits inside `cuMemcpyHtoDAsync` after 97 copies while the main thread sits in the graph launch; `CUDA_DEVICE_MAX_CONNECTIONS=32` and a high-priority stream change nothing [measured]. A launch blocked on a full queue holding a driver lock the copy needs, while the queued graph waits on that copy, fits [inference]; vLLM syncs on each step's sampled tokens, so it queues about one step [inference]. In vLLM (`KSTAGE_AHEAD=1`, laptop, K6 + P1 with one buffer, one draw each): at each MoE layer the next layer's gate, run on this layer's input, names its top `KSTAGE_AHEAD_K` experts and the helper copies those not in a slot. The first three serves hung about 4 s in. Host gdb, two dumps (`results/kv-levers/gdb-p18-ahead.txt`): vLLM's main thread is in `cuMemcpyDtoHAsync_v2` (a torch `setitem` into pageable memory) on a stream that waits for the helper's done flag, and the helper is in `pthread_rwlock_rdlock` inside `cuMemcpyDtoDAsync_v2` [measured]. `ce_lock.sh` puts three blocking calls on the main thread while its stream waits for the helper: with the helper on the caller's context a device-to-host copy and a sync complete and `cuMemFree` deadlocks; with the helper on its own context, or in a child process, all three complete. Beside a kernel on all 82 SMs its 4 GiB copy still runs at 50.9–52.0 GB/s, and the kernel slows 0.5% (caller's context, process) or 2.2% (own context) [measured, `results/kstage-laptop/ce_lock.out`]. So `CE_OWNCTX=1` is the default: the helper makes its own context (259 MiB of device memory, taken from KV) and the done flags live in a VMM page both contexts can write. A process would need shared pages: HOST VMM pages refuse a POSIX-fd handle and HOST_NUMA id 0 takes one [measured, not kept in a results file]. It then serves, ready in 110 s, with helper err 0 in all 66 stats lines. KV 5.44 → 5.15 GiB. Decode, aggregate tok/s at 1 / 4 / 16 streams: without ahead (p17) 169.5 / 235.2 / 348.3; ahead K=6 151.3 / 232.1 / 323.8 (−11 / −1 / −7%); K=10 143.4 / 202.8 / 272.0 [measured]. Misses per layer-step over the whole decode bench fall from 2.96 to 2.17 with K=6 (−27%, copying 3.24 rows a layer-step ahead) and to 1.83 with K=10 (−38%, copying 5.32): about one copied row in four replaces a miss, and the misses fall far less than the 68% / 83% of cold experts k7's prediction names [arithmetic; why is unknown]. At one stream there are 0.19 misses a layer-step, about 20 µs of copy to hide, against +31 µs a layer (5.90 → 6.61 ms a step over 23 MoE layers), so ahead cannot win there [arithmetic]. Eight agents to 127k: 458 s without, 449 s with (decode median 17.9 vs 18.5 tok/s, late TTFT 2.57 vs 2.53 s, misses 3.69 vs 2.59 a layer-step, ahead copying 3.28); P1 with two buffers took 447 s (p15), so a wash [measured]. Answers against p15's K6 + P1: v1 the same scores; v2 held_out2 0.971 vs 1.000 and two_flag 0.9 vs 1.0; alert 6/9 vs 5/9; trap3 noticed 6/12 vs 4/12, 0 fabricated. Items whose score changes: 5 / 3 / 2 (v2 / alert / trap3), against 1–4 / 4–6 / 0 between reruns of one setup, and one of the trap3 gains (rsnyc) is wrong, saying rsync has no `--dry-run`, but scores as noticed [measured]. Where the cost is (`ce2_suite.sh`, `results/kstage-laptop/ce2.out`: ce_probe, 23 layers of ~199 µs work), ms added a step: K=0 / 1 / 2 rows a layer +0.43 / +1.60 / +2.77; with `--devdone` (the graph sets an empty plan's done flag itself, so no round trip to the helper) +0.57 / +1.57 / +2.97; rows on half the layers, K=1 / 2 / 3, +1.46 / +2.05 / +3.54; on a quarter +1.13 / +1.27 / +1.92, and the same with two steps queued [measured]. Above K=0 a dense row costs 44–52 µs (about half its ~108 µs hidden) and a sparse one 66–110 µs, so cutting rows 78% (dense K=1 to a quarter of the layers) cuts the cost only 28% [arithmetic]. The fixed 19–25 µs a layer is most of vLLM's +31 at one stream, and since skipping the round trip does not lower it, it sits in the nodes ahead adds to the graph (plan kernel, flag write, wait), not in the helper [inference]. Where the time goes in vLLM (B1, p19, 2026-10-07; torch profiler on steady decode, `probes/kstage/trace_drive.py` + `trace_ahead.py`, `results/kv-levers/p19-trace.txt`; K6 + P1, K=6, 300-word prompts, one 0.6 s window per arm and stream count; kernels inside CUDA graphs keep their GPU times): the step period without → with ahead at 1 / 4 / 16 streams is 5,731 → 6,421 / 10,873 → 11,825 / 23,707 → 26,114 µs, +30.0 / +41.4 / +104.6 µs a MoE layer. Of that, added kernel time (the union over the graph's two or three streams) is +25.3 / +30.6 / +44.5 and added idle +4.7 / +10.8 / +60.1 [measured]. At one stream the kernels ahead adds a layer are the next gate's fp32 matmul 5.9 µs, `torch.topk` 7.8 (gatherTopK 4.8 + bitonicSort 3.1), the cast of x to fp32 3.5, `_copy_rows` +3.2, `_plan_kernel` 1.6, sigmoid 1.5 and the bias add 1.3: 24.7, against +25.3 [measured]. So the fixed cost is the predictor written as separate torch ops, not the handshake: the flag write and the wait on an empty plan cost at most the 4.7 µs of idle [inference]. The fp32 gate matmul grows with the batch: the router-sized gemmSN goes 16.9 → 33.3 µs a layer at 4 streams and 20.4 → 43.6 at 16, so ahead doubles it [measured]. At 16 streams the helper copies 358 MiB a step (265 copies, 7.9 ms of copy-engine time), yet K6's in-kernel `_copy_rows` falls only 493 → 460 µs a layer (−34), Marlin rises 311 → 335 (+24, probably the helper's copies taking memory bandwidth [inference]), and waits on done flags add 60 µs of idle a layer [measured]. Over each whole serve (warm-up and all three stream counts) misses fall 0.50 → 0.23 a layer-step with 0.38 rows copied ahead, so at most ~71% of copied rows replace a miss [arithmetic; the arms ran different step counts]. At one stream ahead copies 0.35 MiB a step (0.26 copies): nothing to hide, so pure cost [measured]. Under the profiler, per-stream decode was 87.9 / 60.5 / 42.1 tok/s without and 80.3 / 55.0 / 37.6 with (`p19.out`; start and stop stall the server 3–8 s, so these are low at one stream) [measured]. p19's first try died at load with `CUDA_ERROR_OUT_OF_MEMORY` in `cuMemCreate` on an idle card: K6 pins its host pages in 2 MiB pieces and after 11 days of uptime ~34 MiB of free 2 MiB blocks were left; `kv_levers.sh` now drops the page cache and compacts before each phase and logs the free 2 MiB+ blocks (16.6 GiB for p19) [measured]. With the live miss copy (row K; p20, 2026-10-07, two draws), which removes the grid copy's empty floor that p17–p19's ahead arms all paid, ahead K=6 still loses. Decode at 1 / 4 / 16 streams: 178 / 260 / 375 without, 164 / 249 / 353 with (−8 / −4 / −6%). LFU + ahead ran 164 / 285 / 391 against LFU's 176 / 260 / 419, but there ahead's fills still raised the LFU counts [measured]. So the floor was not what hid ahead's gain [measured]. B5 (p21, one draw per stream count, one server each): ahead's fills no longer touch the LFU counts, and new counters track whether each fill is used, evicted unused, or evicts an expert that then misses (`cache_test.py --ahead` checks them against a Python reference on CPU and GPU). Decode with LRU + ahead 158.0 / 245.4 / 320.7, with LFU + ahead 160.6 / 255.6 / 336.8. Of the rows ahead fills, 82 / 72 / 75% are used before eviction under LRU and 33 / 42 / 54% under LFU. Each fill evicts an expert; under LRU those evicted experts later miss 25 / 586 / 10,081 times, 3 / 7 / 32% of the used fills [measured]. At 16 streams LRU + ahead still misses 5.10 a layer-step, where without its fills about 8.4 would have [arithmetic: misses − misses of evicted experts + used fills, approximate]. Under LFU, the experts the predictor names that are not already in a slot are rarer ones, so fewer get used [inference]. Traces with the live copy (p20 trace arms, `results/kv-levers/p20-trace.txt`): at one stream ahead adds +29.5 µs a MoE layer (p19: +30.0), 25.0 of it kernels, so the predictor's separate ops are still the whole fixed cost [measured]. Over each trace serve, ahead halves misses (0.67 → 0.33 a layer-step; 73% of fills used; misses of evicted experts 10% of used fills), and LFU alone cuts them 16% (0.56). Yet per-stream decode at 1 / 4 / 16 streams is 85.9 / 76.0 / 43.1 tok/s without, 79.2 / 68.2 / 41.8 with ahead (−8 / −10 / −3%) and 86.9 / 74.6 / 48.0 with LFU (+1 / −2 / +11%) [measured]. A 0.6 s window cannot measure costs that depend on misses. At 4 streams, live and live + LFU spend 13 vs 55 µs a layer on miss copies. At 16 streams the ahead window ran 4.8% faster (in-kernel copy 449 → 256 µs a layer; idle +79, gate matmul +22, Marlin +12) while its whole streams ran 3% slower [measured]. p24 fused the predictor (`KSTAGE_AHEAD_FUSE=1`, `_ahead_kernel`: the gate dot split over H, sigmoid + bias, top-6 and the plan in one launch up to 16 tokens; in isolation 7.6 vs 16.5 µs a layer at one token and 9–11 vs 34 at 4–16, with the same cache state as the unfused path step for step, `probes/kstage/ahead_fused_test.py`, `results/kstage-laptop/ahead_fused.out`) [measured]. Decode did not move: per-stream 161.1 / 77.4 / 23.4 tok/s at 1 / 4 / 16 streams fused, 160.7 / 77.0 / 22.7 unfused, 180.0 / 78.8 / 26.2 live + LFU [measured]. p25 (B6) split ahead's cost. A dry arm (`KSTAGE_AHEAD_DRY=1`: predictor and flags, every plan empty) runs 158.9 / 75.8 / 25.6 against live + LFU's 180.8 / 76.8 / 25.2, so at one stream ahead's whole 12% loss is fixed plumbing, not copies: 0.76 ms a step, ~33 µs a MoE layer [arithmetic]. GPU-timed waits (`KSTAGE_AHEAD_TICK=1`, `%globaltimer` either side of the flag wait) at one stream: 7.1 µs for an empty plan, 29 µs for a plan with copies, 312 µs a step in all; with the predictor's ~8 µs, ~18 µs a layer is still unplaced [unknown; candidates: the request flag's memop, memop nodes breaking the graph's launch pipelining, the copy launch]. At 16 streams the fixed cost is gone (dry ≈ live + LFU) and the fill waits are the loss: 434 µs a wait with copies, 9.4 ms of a ~43 ms step (22%), while ahead fills ~6.6 experts a layer-step, 43% evicted unused, ~1.8× live + LFU's copy traffic for 28% fewer misses, and 35% of the misses left are experts a fill evicted [measured; ratios arithmetic]. p26 (one draw each) split dry's cost at one stream by leaving parts out (`KSTAGE_AHEAD_SKIP`, dry only). Per-stream decode: live + LFU 174.6 tok/s (its earlier draws 180.0 / 180.8); dry 158.2; without the request memop 159.6; without request and wait 165.6; without the predictor 169.6; without predictor, request and wait 177.7, and the same also without the helper thread 177.1 or with K6's own step as without ahead (ah 0) 177.6 [measured]. Per MoE layer: the predictor ~18 µs (17.7 alone, 18.5 beside the flags), the wait and its reset ~10, the request memop ~2, the helper and ah 0, 30 µs in all, the whole of dry's cost [arithmetic; live + LFU's draws spread 3.5%, ~9 µs a layer, so the request's 2 is within noise]. In isolation the fused predictor takes 7.6 µs a layer, so ~10 µs of its server cost is unplaced [unknown]. At 16 streams, fewer ids predicted a token (`KSTAGE_AHEAD_K`): K = 2 / 3 / 4 / 6 decode 26.8 / 26.7 / 26.2 / 23.3 tok/s per stream (aggregate 387.7 / 385.6 / 380.0 / 342.1) against live + LFU's 25.4 (368.9); misses a layer-step 5.35 / 5.09 / 4.69 / 4.35 against 6.11; fills 1.9 / 3.1 / 4.4 / 6.4 a layer-step, 32 / 36 / 42 / 44% evicted unused [measured; per layer-step arithmetic]. K = 2 is ahead's first win, +5% aggregate, but live + LFU's own 16-stream draws span 25.2–26.2 tok/s, so it needs repeats. The fused kernel passes its state checks at K = 1 and 2 as well (`results/kstage-laptop/ahead_fused_k1.out`, `_k2.out`) [measured]. p27 repeated them (`results/kv-levers/p27.out`): at 16 streams every ahead draw beats every live + LFU draw. Over p26 and p27, live + LFU averages 25.3 tok/s per stream, aggregate 367.2 (draws 364.4–368.9); K = 1 26.4 (381.5, two draws), K = 2 26.7 (385.9, three), K = 3 26.9 (389.4, three, 385.6–392.3): +3.9 / +5.1 / +6.0% aggregate [measured; means arithmetic]. K = 3 fills 3.0–3.1 experts a layer-step, 38% evicted unused, and the experts its fills evict miss 0.74 times a layer-step: 1.89 used fills less 0.74 caused misses = 1.15 saved, against the 1.24 drop measured (6.20 → 4.96) [arithmetic]. At 8 streams K = 2 decodes 47.3 tok/s per stream against 45.1 (+4.8%, two draws each); at 4, 80.6 against 79.7 (+1.1%, about the 1% live + LFU's two draws differ by) [measured]. B4 works as built: with `KSTAGE_AHEAD_MIN=8`, one stream's server made 44 ahead requests (ahead on makes ~5,000) and decoded 174.5 tok/s against live + LFU's 180.5 in the same round; live + LFU's one-stream draws span 174.6–180.8, so one draw cannot say whether anything is left [measured]. K = 3 from 8 tokens is the first ahead setting with no measured loss. p28 traced live + LFU, dry and K = 3 at 1 / 4 / 16 streams (`results/kv-levers/p28-trace.txt`, `probes/kstage/trace_overlap.py`). p26's unplaced ~10 µs is the fused predictor itself, slower in the server than alone: a median 16.0 / 17.3 / 18.7 µs a call (dry; 7.6 alone), while idle time grows 1.8 / 3.7 / 3.9 µs a layer [measured]. It runs on the routed experts' stream straight after `_copy_rows_live` (0.1 µs apart), so it lengthens that branch; the other branch's kernels (dense Marlin, LoRA) run during 12.3 µs of each call at one stream but 4.0 at 16 [measured], so at one stream it shares the SMs and at 16 it does sixteen tokens' work [inference]. K = 3 adds 18.1 / 22.4 / 22.5 µs a layer of predictor and 3.0 / 5.7 / 8.3 of idle, so its fill waits are now small; its copy engine moves 6.5 / 41.5 / 165.8 MiB a step beside the compute [measured]. The 16-stream windows are 30 steps, too few to split the live copy: dry, which fills nothing, cut `_copy_rows_live` by 71.6 µs a layer against K = 3's 73.8, while p25's one dry draw at 16 streams decoded 371.7, 1.2% over live + LFU's mean [measured], so the throughput draws above stay the measure of K = 3. Not written off. p29 ran K = 3 from 8 tokens on the load it is for, against live + LFU, one draw each: 8 agents to 127k took 418 s, 20.1 tok/s a stream, late TTFT 2.49 s, against 424 s, 19.6, 2.52 s; 16 agents to 127k with `kvoff16` 816 s, 10.5 tok/s, 4.92 s, 17 preemptions, against 817 s, 10.3, 4.71 s, 13 [measured]. Live + LFU drew 418 s at 8 agents in p22, so on the agent load ahead neither gains nor costs [measured]. At one stream the pairs drew 175.7 / 179.0 against 180.4 / 180.7: K = 3 from 8 tokens has trailed in all three same-round pairs (mean 176.4 against 180.5, −2.3%) though its one-stream graphs plan nothing [measured]. But p26's arm with every ahead part skipped, which is what the one-stream graphs run here, drew 177.6 against live + LFU's 174.6, and its spinning helper cost nothing measurable (177.1 without, 177.7 with), so the four pairs split 3–1 and the gap is unresolved [measured]. At 8 streams K = 3 decoded 48.8 tok/s a stream (328.4 total, one draw) against live + LFU's 45.2 / 45.0 and K = 2's 46.9 / 47.6 in p27, +8.2% over live + LFU [measured]. p30 let a fill evict only empty slots and experts below U uses' worth of LFU count (`KSTAGE_AHEAD_STALE=U`; the 0.74 misses a layer-step K = 3's evictions cause), two draws each at 16 streams: U = 2 drew 370.0 / 366.8 against live + LFU's 373.7 / 369.4, giving up ahead's whole gain; U = 32 drew 387.8 / 391.0, K = 3's own level (389.2 this round, 389.4 over p27's three); U = 8 drew 379.3 / 401.3, a spread six times any other arm's [measured]. So far, limiting what a fill may evict has not paid: the misses its evictions cause cost less than its fills save [inference]. K = 3 from 8 tokens decoded 48.4 tok/s a stream at 8 streams (p29: 48.8) [measured]. Every side-stream arm (`KSTAGE_AHEAD_SIDE=1`) died in graph capture: plan() skips the last MoE layer, but join() still made the capturing stream wait on the uncaptured side stream (fixed in ae35042). p32 (two draws an arm, c=16): live + LFU 366.2 / 372.2, K3-min8 389.3 / 387.8 (p27's 389.4 again), side 384.0 / 381.7, stale8 377.0 / 374.9, side + stale8 378.9 / 375.8 [measured]. Hiding the plan on a side stream loses 1.5% and 0.05 GiB of KV (5.16 → 5.11); at 8 streams it gives 47.7 a stream against K3-min8's 48.4–48.8, and 8 agents to 127k 415 s against 418 (p29). U = 8 over four draws averages 383.1, below K3, with p30's 401.3 the outlier. The side + stale8 server answers as K6 does: signed v1 129, v2 116, alert 4, trap3 4, the same items as p15's K6 on 158 of 158, 132 of 134, 8 of 9, 12 of 12. At one stream K3-min8 is neutral: p31's four pairs give 174.6 against 175.7, ahead leading two [measured]. K3-min8 on the main stream stays the best configuration; the side stream and stale-only fills are closed. Next: live misses copied by the copy engine from 8 tokens (48.5–50.9 GB/s against the live kernel's ~39 alone, `copy-rows-probe.txt`), and one request per two layers (B3; two layers ahead names 58% / 72%; less needed now that the waits are small). |
| P | Compute cold experts on the CPU | `probes/kstage/cpu_expert.c`: NVFP4 read as the checkpoint stores it, AVX2 + FMA (no AVX-512), a pool of spinning threads; `cpu_expert_probe.py` | correct: max error 1.3e-7 of the largest output against a float64 dequant, 1–24 threads, and vLLM's own e2m1 table matches. Laptop, idle, experts cycled through 2 GiB so every call reads RAM: all-thread RAM read 28.8–29.3 / 71.5–72.2 / 90.3–90.8 / 94.1–94.7 / 68.7–76.9 GB/s at 1 / 4 / 8 / 16 / 24 threads (two runs, 2026-10-05 and -06). An expert at one token takes 646–856 µs on one thread, 166–254 on 8, 115–175 on 16 (32–49 GB/s), 87–3,343 on 24 (erratic). Four tokens an expert 280–285 µs, 16 tokens 1,125–1,155 µs (16 threads) [measured] | measured: no faster than the copy engine (~110 µs an expert at 51.8 GB/s), with every core spinning. The earlier ~75 µs guess assumed the kernel reads at the RAM's rate; it reaches half. Its one use is beside the copy engine: both read the same RAM, 94 GB/s against the copy engine's 51.8, so splitting misses could at most raise the miss rate 1.8x [arithmetic]. It needs the same graph-to-host handshake as row O, which works (with the same cap on queued steps). Prefill is out: an expert there serves ~96 tokens |
| Q | More streams | `--max-num-seqs 64`, decode at 16 / 32 / 64 | gather4: 191 / 245 / 357 (KV 4.66 GiB); with the copy engine from 32 tokens (`KSTAGE_DMA_M=32`) 194 / 267 / 447. No offload: 669 / 691 / 769 (1.40 GiB; 23 running, 41 waiting at 64). With DMA, offload's share of no offload rises 0.29 → 0.39 → 0.58 | done |
| R | Async scheduling | `--async-scheduling` | gather4 98 / 141 / 192, UVA 77 / 97 / 138: the same as without. vLLM 0.29.0 already turns it on here | done: already the default |
| S | The adapter under kstage | G6q LoRA through each mode | speed below; the plugin now hands MoE LoRA the router's ids. Research eval under K2 gather = under UVA: v1 0.989, v2 0.986; alert 6 vs 4 of 9, trap3 4 vs 5 of 12 (too few items to separate) | done (G6q has no routed-expert LoRA; fix for adapters that do). K6 (round 11): v1 0.989, v2 1.000, alert 5 of 9, trap3 4 of 12: K2's within the noise |
| T | Decode graphs without torch.compile | `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'`; `VLLM_USE_BREAKABLE_CUDAGRAPH=1` with mode 0 (piecewise graphs, experimental) | Eager's 1.40 GiB of extra KV (row H) is 0.95 of peak activation, measured in a profile pass through the compiled model, plus 0.45 of non-torch memory; the graphs themselves take 0.09. Round 12, `FULL_DECODE_ONLY` with mode 0 (`-fdo`): no offload KV 1.69 → 2.56 GiB, decode 202 / 430 / 701 → 188 / 359 / 712, 98k 15.4 → 15.5 s; UVA 4 GiB 5.63 → 6.50, 76 / 98 / 132 → 75 / 98 / 140, 34.8 → 34.4 s; K6 5.71 → 6.59, 166 / 241 / 294 → 156 / 203 / 297, 25.6 → 25.8 s. Breakable graphs under UVA (`-brk`) crashed in warmup: an illegal memory access surfaced in the grammar-bitmask kernel after both captures finished; the faulting kernel is unknown (asynchronous error) | `-fdo` done: +0.87 GiB KV on every arm; costs 0–7% of decode at one stream and 16 streams but 16–17% at four (K6's c=4 swings 211–259 between identical runs, so that is one draw); adopting it is David's call. `-brk` crash not localized (needs `CUDA_LAUNCH_BLOCKING=1`, or a run without offload) |
| U | K6 prefill staged on the copy engine (P1) | `KSTAGE_DMA_M=64`: batches of 64+ tokens outside CUDA graphs leave the cache alone. Every non-resident expert of the layer `KSTAGE_DMA_BUF` ahead (default 2) is copied by the copy engine into staging rows, so Marlin reads no host memory | the cost it attacks: 98k prefill 25.3 s with K6 vs 15.4 s without offload. Copying the ~4 GiB of cold experts once per 2,048-token chunk at 51.8 GB/s is ~83 ms against K6's ~206 ms extra a chunk [arithmetic]. Staging takes ~0.5 GiB of device memory for two buffers [arithmetic]. Desktop, k13b, one draw: the first build crashed at install (fixed). The second sorted every expert tensor as small, so all staging ran as an on-device gather by SMs instead of the copy engine: prefill (7,560 tokens) 3.51 s with 64 slots or all, against K6 alone's 3.23 s (64 slots) and 2.70 s (all) and 1.14 s without offload. Decode is K6's own (126–131 / 162–195 / 212–234 vs 127–135 / 179–194 / 202–228): decode batches stay under 64 tokens. The MoE layers of a step took 501–528 ms at 80–244 tokens and 751–766 ms at 2,048, so gathering the ~765 cold experts (~4 GiB) costs ~500 ms a step, ~8 GB/s [arithmetic]. Greedy outputs match no offload 4–5 of 8, the permutation's pattern. Laptop, p15, fixed build, one draw each: 98k prefill 18.00 s (8k 0.97, 32k 4.64), against K6's 25.3 and no offload's 15.4; at `--max-num-batched-tokens 8192` 13.64 s (KV 4.36 GiB). The copy engine is busy 87–91 ms a 2,048-token step for 4.00 GiB (47–49 GB/s) while the step's MoE layers span 417–443 ms; at 8,192 tokens 84–86 ms (50–51 GB/s) against 577–1,277 ms. Steps of 80–208 tokens are copy-bound: 87–89 ms of copying against a 99–112 ms span. KV: two buffers 5.17 GiB, one (`KSTAGE_DMA_BUF=1`) 5.44, K6 alone 5.71; one buffer's 98k is 17.67 s. Decode is K6's (batches under 64 tokens): 168 / 238 / 351 aggregate (b8k 168 / 271 / 371). Research eval: v1 0.989, v2 1.000, alert 5 of 9, trap3 4 of 12, K6's scores exactly; v1 52 s, v2 47 s against K6's 93 s and 83 s [measured] | measured on the laptop: P1 removes 74% of K6's prefill penalty, (25.3 − 18.0) / (25.3 − 15.4) [arithmetic]; at batch 8,192 it beats no offload at the default batch. The remaining 2.6 s on a 98k prefill, ~54 ms a chunk against ~89 ms of copying, is not explained [unknown]; copies sharing the card's memory with Marlin's reads would fit [inference]. A higher `KSTAGE_DMA_M` changes nothing here (p16, one buffer, one draw each): 128 gives 98k 17.53 s, decode 167 / 248 / 346; 256 gives 17.53 s, 165 / 258 / 353; 64 gave 17.67 s, 167 / 273 / 341 [measured]. The bench has no steps between 64 and 256 tokens (2,048-token chunks, decode at 16 or fewer), so the threshold stays untested where it would matter: the agent loads' mixed steps [inference]. Desktop rerun with the copy engine (k13c, one draw each): prefill (7,560 tokens) 2.40–2.42 s, against the SM gather's 3.51–3.52 (k13b), K6 alone's 2.70–3.23 and no offload's 1.14. The copy engine moves 4.00 GiB a step in 354–369 ms (11.6–12.1 GB/s, the desktop's 12.2 ceiling, row L), and a 2,048-token step's MoE layers span 505 ms against the gather's 751–766, so on the desktop P1 is bound by its copy. Decode stays K6's, within a draw of the gather: 126 / 165 / 211 (64 slots), 131 / 212 / 218 (all), 135 / 206 / 222 (all, one buffer) against 126 / 162 / 212, 131 / 195 / 234, 130 / 194 / 233. Against both references the copy engine's greedy outputs (all slots) show the same first differences and drift as the gather's [measured] |
| V | A different MoE kernel (p38) | `KSTAGE_MOE_NOLORA=1` (`-lgate`) lets vLLM pick a kernel without LoRA support; `--moe-backend flashinfer_b12x` (`-moeb12`) with `KSTAGE_B12X_FIX=1`, and `KSTAGE_B12X_EXACT=1` for exact scales | FlashInfer's B12x prefills 98k 21% faster (11.66 vs 14.84 s) but takes 0.75 GiB of KV in agent arms; agents to 127k 351 → 381 s (8), 697 → 781 s (16). Marlin pinned to B12x's KV: 389 / 792 s. The CUTLASS backends refuse the checkpoint | measured: not adopted. The agent loss is all KV; at equal KV the kernel gains 1–2%, so the agent wall is not mostly MoE prefill |

Sourced facts these options rest on (vLLM 0.29.0 source):

- **UVA** (`--offload-backend uva`, the default) maps pinned host memory into the GPU's
  address space. The GPU reads offloaded weights over PCIe as kernels run, and reads only
  what it touches. Decode at one stream touches 6 of 128 experts a layer [sourced; the
  traffic estimate is arithmetic].
- **Prefetch** offloads layers whose index mod G is at least G − K, over all 52 layers,
  and skips layers with nothing matching `--offload-params`. With G = 4, 2, 1 that is 7,
  13 or all 23 MoE layers. A static pool of `--offload-prefetch-step` slots holds the
  layers in flight, one layer's experts (~0.67 GiB) per slot. It copies whole layers, and
  is written for torch.compile and CUDA graphs [sourced; MoE counts are arithmetic].
- **Prefetch races when the step does not divide the layer count.** Layer i reads slot
  i mod step and, once done, copies layer (i + step) mod n into it; a layer waits only on
  its own copy. With n mod step ≠ 0 a wrap-around copy overwrites a slot that a later
  layer in the same pass still reads [sourced, `offloader/prefetch.py`]. On the desktop,
  G4/K1 (n = 7) at step 2 flipped the first token of 2–3 prompts in 8, differently each
  run; step 1 matched 8 of 8 [measured]. Race-free for Lightning: G4/K2 (n = 12) at steps
  1–4, G3/K1 (n = 6) at 1–3, G4/K1 at step 1 only [arithmetic]. `KSTAGE_PFFIX=1` defers
  the wrap-around copies (CPU model: `probes/kstage/pffix_test.py`).
- **Async scheduling is already on.** With `--async-scheduling` unset, vLLM turns it on
  unless the model pools, the drafter is incompatible or the executor cannot [sourced,
  `vllm/config/vllm.py`]; none applies here.
- **The prefill batch** defaults to 2,048 tokens for an API server on a card under 70 GiB
  [sourced, `vllm/engine/arg_utils.py`]. Under UVA each batch reads every expert a layer
  routes to, so a larger batch reads the same bytes for more tokens [inference].
- **Prefix hits land on 4,176-token boundaries** (the attention block). The agent's turns
  recompute up to 4,175 tokens each [measured, p6: 4,176 of 8,192 cached on a repeat].
- **Host KV can deadlock the scheduler.** A host KV load is always asynchronous: the request
  is admitted with blocks for its loaded prefix plus a reservation for the rest of its length
  (`scheduler_reserve_full_isl`, on by default), and keeps both while it waits. It starts
  only when the waiting loop reaches it; the loop stops at the first request it cannot
  admit, and only running requests are preempted [sourced, `v1/core/sched/scheduler.py`].
  p34 caught it at 8,192-token chunks (73 GPU blocks): 32 free, 12 reserved. The queue
  head, a preempted request, needed 22 of the 20 the reservation left; behind it two
  requests with 66,816 of 73,472 tokens loaded held their blocks and needed 6 each. The
  snapshot was unchanged 600 s later [measured; which check failed is inference]. No 0.29.0
  setting avoids it: the reservation is already on and the watermark is 0 [sourced,
  `vllm/config/scheduler.py`]. `KSTAGE_UNJAM=1` (`probes/kstage`) rotates each waiting queue
  once per step that has scheduled nothing for 2 s. It only reorders, so no block is freed
  under the connector. In p34 all three stalls cleared after one step and both arms finished
  [measured]; answer quality under it is unchecked. `KSTAGE_UNJAM=diag` logs the queues.

### K: hot experts on the card (`probes/kstage`)

Stock offload picks whole layers: `--cpu-offload-gb 4` moves every routed expert of the
first 6 MoE layers, hot or cold, and Marlin reads them through UVA as it runs. Routing is
skewed (row J), so the plugin keeps the experts that are routed to most on the card and
moves only the rest. It is a vLLM general plugin, mounted on `PYTHONPATH` and switched by
`KSTAGE=<mode>`; with `KSTAGE` unset, `register()` returns before touching vLLM.

| Mode | What it does | Off the card at 4 GiB |
|---|---|---|
| K2 `gather` | Stock offload's layers, but before each offloaded MoE runs, the experts the batch routes to are copied into a 0.67 GiB device buffer (a Triton gather, or the copy engine per row from `KSTAGE_DMA_M` tokens), and Marlin reads the buffer | 6 whole layers |
| K5 `hotcold` | Each layer's expert tensor is one CUDA VMM range of device pages (hot rows) and host pages (cold rows). Routing ids are remapped so the cold experts sit at the end; the cold set is the least-routed layer/expert pairs anywhere (`KSTAGE_POLICY=global`, from a routing profile). Cold rows are read in place through UVA | 765 experts across 23 layers |
| K6 `cache` | Hotcold's split, but the device rows are a cache: H pinned to the hottest experts, S slots (`KSTAGE_SLOTS`, default all) that take whatever the batch routes to and evict the least recently used. Misses are copied home → slot before Marlin runs | the same 765 |
| `predict` | A probe: how well each layer's input, through the next layers' gates, predicts what they route to. Decides whether K6 slots can be filled ahead | none |

**Laptop**, 131,072-token server, gmu 0.92, `offload_bench.py` (decode at 1 / 4 / 16
streams, aggregate tok/s; cold prefill in seconds):

| Arm | KV GiB | Decode | Prefill 8k / 32k / 98k |
|---|---|---|---|
| No offload | 1.69 | 202 / 430 / 701 | 0.78 / 3.78 / 15.40 |
| UVA, 4 GiB | 5.63 | 76 / 98 / 132 | 2.29 / 10.15 / 34.75 |
| UVA, 4 GiB, prefill batch 8,192 | 4.82 | 77 / – / 141 | 1.87 / 7.89 / 27.19 |
| K2 gather, 4 GiB | 4.95 | 99 / 141 / 193 | 1.43 / 6.61 / 23.52 |
| K2 gather, 4 GiB, DMA | 4.95 | 99 / – / 199 | 1.19 / 5.64 / 20.92 |
| K5 hotcold, 4 GiB | 5.74 | 110 / 173 / 232 | 1.52 / 6.87 / 26.67 |
| K2 gather, 8 GiB | 8.89 | 65 / 79 / 114 | 2.09 / 9.57 / 32.18 |
| K5 hotcold, 8 GiB | 9.75 | 60 / 80 / 103 | 2.74 / 12.25 / 44.71 |
| K2 gather, 16 GiB | 16.15 | 39 / 45 / 65 | 3.29 / 14.92 / 47.97 |
| K2 gather, 4 GiB, DMA, batch 8,192 | 4.14 | 101 / – / 207 | 0.85 / 3.82 / 15.10 |
| K5 hotcold, 4 GiB, DMA | 5.06 | 111 / – / 217 | 1.71 / 7.63 / 27.06 |
| K5 hotcold, 4 GiB, DMA, batch 8,192 | 4.25 | 111 / – / 246 | 1.12 / 4.89 / 18.19 |
| K6 cache, 16 slots a layer | 5.73 | 155 / 215 / 295 | 1.58 / 7.04 / 25.99 |
| K6 cache, 64 slots | 5.72 | 165 / 211 / 305 | 1.69 / 7.30 / 25.34 |
| K6 cache, all slots (LRU) | 5.71 | 166 / 241 / 294 | 1.77 / 7.53 / 25.56 |
| K6 all slots, LFU | 5.72 | 166 / 257 / 340 | 1.72 / 7.46 / 25.37 |
| K6 all slots, `KSTAGE_COPY=82` | 5.71 | 175 / 259 / 314 | 1.81 / 7.43 / 25.26 |
| K6 + P1 (row U), all slots, LRU (p20, 2 draws) | 5.44 | 166 / 258 / 343 | – |
| the same, `KSTAGE_COPY=82` | 5.44 | 177 / 245 / 365 | – |
| the same, `KSTAGE_COPY_LIVE=82` | 5.44 | 178 / 260 / 375 | – |
| the same, `KSTAGE_COPY_LIVE=82` + LFU | 5.44 | 176 / 260 / 419 | – |

6 agents growing to 98k: no offload 730 s, UVA 572 s, K2 gather 402 s (decode median
15.2 tok/s, late time to first token 2.63 s), prefetch with the fix 650 s, K6 282 s
(23.0 tok/s, 2.68 s), K6 LFU 271 s (25.0 tok/s, 2.49 s) [measured].

- **K6 on the laptop:** 2.2x UVA's decode at one stream and 2.2x at 16, with 5.71 GiB of
  KV against UVA's 5.63, and half UVA's agent wall time. Host RAM: +10.4 GiB with 16 slots,
  +16.3 GiB with 64, +20.0 GiB with all (no offload +4.1 GiB) [measured].
- **Its weak spot is prefill:** 25.3 s at 98k against 15.4 s with no offload. A 2,048-token
  chunk makes 12,288 routings a layer, enough to reach most of its 128 experts, so each
  chunk misses and copies most of the 765 cold ones [inference]. P1 (row U) moves those copies to the copy engine.
- **Slot count barely matters on the laptop** (16 vs all: 295 vs 294 at 16 streams), unlike
  the desktop (127 vs 236): the laptop's link copies a miss 4x faster [inference].
- **LFU vs LRU, `KSTAGE_COPY`:** LFU 340 vs LRU 294 at 16 streams and 271 vs 282 s on the
  agents; `KSTAGE_COPY=82` 314 at 16 streams and 175 at one. Single draws each.
- **The miss copy's empty launch (p20, 2026-10-07):** the default `_copy_rows_kernel` runs
  a program per 512 words of every *possible* copy (P = the step's routings, up to 96 at 16
  streams), live or not. Alone, in CUDA graphs, with nothing to copy it costs 8 / 31 / 224 µs
  at P = 6 / 24 / 96 for one 2.38 MiB w13 tensor; `KSTAGE_COPY=82` (82 programs striding
  the plan) 1.0 / 2.1 / 11.9; the new `KSTAGE_COPY_LIVE=82` (count the live copies, walk
  only those) 0.9 / 0.9 / 1.5. A real row costs ~64 µs (~39 GB/s) in all three
  (`probes/kstage/copy_rows_probe.py`, `results/kv-levers/copy-rows-probe.txt`) [measured].
  In serving, live takes K6 + P1 from 166 / 258 / 343 to 178 / 260 / 375, and with LFU to
  176 / 260 / 419 at 16 streams, the best K6 decode yet (2 draws each, `p20.out`). The
  profiler agrees: the copy kernel's time a layer falls 33.7 → 7.4 µs at one stream (step
  −7%) and 150 → 13 at four (step −23%); at 16 it is 493 → 449 µs, now nearly all real
  copies, ~45% of the 23 ms step (`trace_ahead.py trace-p19-trace-base
  trace-p20-live-trace`) [measured]. So at 16 streams the lever is fewer misses: over each
  whole serve LFU makes 2.43 a layer-step against LRU's 3.07 (−21%) [measured, one serve
  each]. p17-p19's ahead and DMA arms all ran the grid copy.

- **Both modes beat stock offload** at every size. At 4 GiB, hotcold has the faster
  decode (232 vs 193 at 16 streams) and gather the faster prefill (20.9 s with DMA vs
  26.7 s at 98k).
- **At 8 GiB gather wins both.** Inference: by then the cold set holds experts that are
  routed to often. Hotcold reads each of them in place at UVA's ~27 GB/s every time, while
  gather copies them at 37–51 GB/s and Marlin then reads device memory.
- **Gather's buffer costs KV:** 0.67 GiB, so 4.95 vs hotcold's 5.74 GiB.

**Desktop** (RTX 3090; `kstage_client.py`: decode over short "Story" prompts, prefill
7,560 tokens). Its link reads at 7.2 (UVA) to 12.2 (copy engine) GB/s, a quarter of the
laptop's, so offload costs more there:

| Arm | KV GiB | Decode | Prefill (s) | Host RAM GiB |
|---|---|---|---|---|
| No offload | 0.69 | 204 / 465 / 631 | 1.14 | |
| UVA, 4 GiB | 4.63 | 28 / 37 / 55 | 6.06 | 4 |
| K2 gather | 3.95 | 45 / 61 / 92 | 2.82 | 4 |
| K5 hotcold, profile | 4.73 | 31 / 43 / 67 | 5.93 | 4.06 |
| K6 cache, 16 slots a layer | 4.73 | 94 / 122 / 127 | 4.72 | 5.95 |
| K6 cache, 64 slots | 4.71 | 123 / 170 / 212 | 3.14 | 11.70 |
| K6 cache, all slots (LRU) | 4.71 | 130 / 178 / 236 | 2.62 | 15.38 |

- **K6 is the first mode that changes the picture:** plain LRU decodes 4.3–4.8x stock
  offload with the same KV, and prefills in 2.62 s against 6.06 [measured]. In each busy
  30-second window 0.4–1.7 routings a layer-step missed (0.89 with all slots), of 96 a
  layer-step at 16 streams: about 1% [arithmetic].
- **Its cost is host RAM, not VRAM:** every expert that can be evicted keeps a home row in
  pinned host memory, 15.38 GiB with all slots (the desktop has 30 GB).
- **The caveat:** 16 streams of near-identical prompts route alike, which flatters a cache.
  The laptop's 6-agent run (`p10-ks-cache4-a6`) is the test that counts.

**With the G6q adapter** (laptop, `kstage_client.py`; do not compare its decode with the
first table, whose prompts are 1k tokens):

| Arm | Decode | Prefill 7,560 tokens (s) |
|---|---|---|
| No offload | 196 / 492 / 923 | 1.13 |
| UVA, 4 GiB | 75 / 110 / 183 | 2.45 |
| K2 gather | 98 / 150 / 254 | 1.62 |
| K2 gather, DMA | 97 / 152 / 256 | 1.46 |
| K5 hotcold | 86 / 136 / 232 | 2.41 |

With the adapter, gather beats hotcold at 4 GiB too; the cause is unknown.

**MoE LoRA and remapped ids.**
- **The problem:** vLLM 0.29.0 picks each routed expert's adapter by the ids Marlin gets
  (`apply_w13_lora`; w2 reuses its alignment) [sourced, `marlin_moe.py`]. Hotcold permutes
  those ids, and K6 sends ids up to E + slots, past the adapter's 128 experts. Small
  decode batches do not filter them out (`naive_block_assignment`).
- **G6q is unaffected:** it has no routed-expert LoRA, so that stack is zero.
- **The fix:** the plugin hands the LoRA call the router's own ids.
  `probes/kstage/lora_fix_test.py` checks it on the CPU, and fails with the fix disabled.

**Checking outputs, not just speed.**
- **The method:** `kstage_client.py` saves 8 prompts × 96 greedy tokens, with the top 2
  logprobs at each position. `lpcmp3.py` compares an arm with a reference run: full
  matches, the first differing token, the reference's margin there, and the largest
  logprob drift before it. Two no-offload runs on the desktop match 7 of 8 prompts.
- **Same order, same output:** stock UVA, gather and hotcold with experts in index order
  match no offload 8 of 8.
- **Permuted rows diverge:** hotcold with a profile matches 3 of 8, and K6 3–5. Their
  first-token logprobs are within 0.06 of no offload. The racy prefetch flips first tokens
  and is off by 0.5–4.5.
- **The cause is the permutation, not the cold reads** [measured, k9]: hotcold's
  profile permutation with nothing cold (`hc-perm0`) matches no offload 3 of 8, with the
  same first differing tokens as hotcold with 4 GiB cold, and the two match each other
  8 of 8. Each divergence starts where the reference's top two tokens are within
  0.00–0.25 of each other. That a permuted layout changes the order Marlin sums in is
  still an inference.
- **Where K6's decode time goes** [measured, k12, desktop, one draw each]. Decode at
  1 / 4 / 16 streams, tok/s:

  | Arm | What it keeps | Decode | Prefill (s) |
  |---|---|---|---|
  | No offload | - | 205 / 465 / 634 | 1.14 |
  | `cold0-frz1` | cache kernel and remap, 16 slots, no copy launch | 203 / 468 / 627 | 1.14 |
  | `cold0-s16` | the same plus the (empty) copy launch | 195 / 451 / 614 | 1.14 |
  | `cold0-all` | the same with every slot | 194 / 443 / 579 | 1.14 |
  | K6, slots all | real host traffic: 1.54 misses, 1.50 copies a layer-step | 135 / 194 / 228 | 2.70 |
  | `cache4-all-frz1` | no copies: every miss (4.89 a layer-step) read over UVA | 30 / 42 / 68 | 6.28 |

  `cold0` puts nothing in host memory, so nothing can miss; `frz1` freezes the slots, so
  nothing is copied. At one stream the machinery adds 0.26 ms a token (4.89 to 5.15 ms),
  0.19 ms of it the copy step `frz1` skips, and K6 adds 2.51 ms: about 10% machinery, 90%
  host reads and copies [arithmetic]. Copying misses into slots instead of reading them
  over UVA is worth 4.5x on the desktop's 12.2 GB/s link. All five arms match no offload's
  greedy output on the same 3 of 8 prompts with the same first differing tokens, the
  permutation's pattern: the UVA reads are exact.

**Copy rates, laptop** (GB/s, `x-copy` probe): device memory 145.5; Marlin's read through
UVA 27.2; copy engine, whole range 51.6, per row 48.5–50.9; Triton gather 36.8; a VMM
range's host half 26.7. On the desktop: 7.2 / 12.2 / 11–12.5 [measured].

### V: a different MoE kernel (p38)

Every memory lever above stops at the prefill-compute floor (1.43M uncached tokens for eight
agents, 2.86M for sixteen), so p38 asked whether a faster expert kernel lowers the floor.
Lightning's experts run on Marlin, a W4A16 kernel: it stores weights as FP4 and multiplies
them in BF16.

- **Why it is always Marlin** [sourced]: with an adapter loaded, vLLM 0.29.0's LoRA gate
  (`modular_kernel.py:594`) rejects every fused-MoE kernel that has no LoRA support. G6q puts
  no LoRA on the routed experts (row S), so the gate guards nothing here.
  `KSTAGE_MOE_NOLORA=1` (suffix `-lgate`) skips it, and `-moeb12` / `-moefic` / `-moecut`
  pass `--moe-backend`.
- **CUTLASS refuses the checkpoint** [measured]: both FLASHINFER_CUTLASS and VLLM_CUTLASS
  stop at load with "does not support quantization scheme QuantKey(u8,scale(f8e4m3fn,static,
  GroupShape(row=1, col=16)),scale2(f32,static,per_tensor),symmetric)xNone". The
  checkpoint is `quant_algo=W4A16_NVFP4`.
- **B12x needs two fixes to load** (FlashInfer's FP4 MoE for sm_120, `KSTAGE_B12X_FIX=1`).
  First, the stored weights pad the expert width from 1,856 to 1,920, but vLLM gives the
  kernel 1,856, so it crashes on shapes. The fix reads the width from the scale tensor.
  Second, each of the 24 MoE layers allocates its own workspaces and output buffer, which
  runs out of memory. Layers run one at a time, so the fix shares those buffers per shape.
  The padding costs 0.45 GiB of weights, and peak activation is 0.30 GiB higher [measured].
- **B12x runs W4A4, not W4A16** [sourced]: vLLM builds the kernel without a `quant_mode`,
  and FlashInfer then defaults to `nvfp4`, which rounds activations to FP4 per 16-value
  block inside the kernel. FlashInfer's W4A16 mode keeps a second, repacked copy of every
  expert, and this card has no room for one.
- **vLLM's hand-off to B12x loses precision** [sourced + measured]: it multiplies each
  expert's global weight scale (6e-5 to 1e-3 here) into the FP8 E4M3 block scales and
  sets the global scale to 1. That pushes the block scales into FP8's subnormal range,
  where it has fewer bits. Six sampled layers show a median block-scale error of 5–7%,
  a p99 of 18–29%, and 0–0.55% of blocks rounded to zero (`probes/b12x_scale_bake.py`).
  Marlin receives the global scales separately, so it is exact. `KSTAGE_B12X_EXACT=1`
  (`-exact`) keeps the stored block scales and passes the global scales to the kernel as
  its per-expert alphas, with the activations' global scale set to 1 as before. It costs no
  speed (decode 184.3 / 645.2 tok/s, 98k 12.13 s, against 184.5 / 644.3 and 12.15 baked),
  and the eval cannot tell the two apart (below) [measured]. The size of the difference in
  the logits is unmeasured. If B12x is ever used, `-exact` is the correct setting, since it
  matches the checkpoint's scales [sourced].

Speed, one draw each (98k = a 98,304-token prompt, cold):

| Arm | Weights (GiB) | KV | Decode c=1 / c=16 (tok/s) | Prefill 8k / 32k / 98k (s) |
|---|---|---|---|---|
| Marlin, no LoRA | 17.86 | 1.79 GiB, 488,243 tokens | 221.7 / 718.4 | 0.77 / 3.64 / 14.84 |
| B12x, no LoRA | 18.3 | 1.37 GiB, 373,555 tokens | 199.6 / 565.0 | 1.96 / 2.62 / 11.66 |
| B12x + G6q | 18.33 | 1.27 GiB, 347,340 tokens | 184.5 / 644.3 | 2.05 / 2.78 / 12.15 |

B12x prefills 98k 21% faster than Marlin and 32k 28% faster [arithmetic]. With the adapter
loaded on both, B12x is also 21% faster (12.15 s against 15.4 s). Its cold 8k probably
includes a first-call compile [inference]: warm 8k takes 0.26 s against Marlin's 0.39.
Decode is 10% slower at one stream and 21% slower at sixteen, both without the adapter
(199.6 / 565.0 against 221.7 / 718.4 tok/s, one draw) [arithmetic].

Agents to 127k, prefix KV in RAM, 4,096-token chunks, G6q loaded. The B12x arms add
`KSTAGE_UNJAM=1`, which acts only on a stall, and no 4,096-chunk run has stalled:

| Arm | GPU KV | 8 agents: wall / late TTFT | 16 agents: wall / late TTFT / preemptions |
|---|---|---|---|
| Marlin (p33/p35) | 1.8 GiB | 351 s / 8.27 s | 697 s / 22.55 s / 0 |
| B12x (p38c, p38d) | 1.05 GiB | 381 s / 12.50 s | 781 s / 28.31 s / 51 |
| Marlin with KV pinned at 1.05 GiB (`-kvb105`, p38f) | 1.05 GiB | 389 s / 12.94 s | 792 s / 29.56 s / 51 |

Both kernels compute the same 1.43M and 2.86M uncached tokens. B12x's KV pool is 0.75 GiB
smaller: 0.45 GiB of padded weights (18.33 against 17.88 GiB) and a 0.30 GiB higher
activation peak at 4,096-token batches (1.32 against 1.02) [arithmetic]. At sixteen agents
the pool holds about two 127k contexts (285,081 tokens), so requests are preempted. Pinned
to the same 285,081 tokens, Marlin runs the same schedule as B12x (51 preemptions, identical
prefix and RAM hits) and takes 389 s and 792 s. At equal KV B12x is 2% and 1.4% faster
[measured, one draw], so its whole agent loss is the KV it costs. The kernel gain is also
far below what the 98k prefill predicts: saving 32 µs a token, as it does there, would take
46 s off eight agents and 93 s off sixteen [arithmetic], and 8 s and 11 s came off. The
agent wall is therefore not mostly MoE prefill [inference]. Where else it goes (13.6M /
27.1M tokens of KV loaded from RAM, attention over long contexts, decode, at which B12x is
slower) was measured next: attention ("L: where the agent wall goes"). The same
arithmetic rules out porting K6 to B12x for agents: winning
back the 0.75 GiB would leave the 1–2% kernel gain [inference].

Answers, G6q (thinking off; Marlin row from `tm-graphs092`):

| Arm | v1 hit and grounded | v2 grounded / hit and grounded | alert correct | trap3 noticed |
|---|---|---|---|---|
| Marlin | 0.989 | 1.000 / 1.000 | 0.556 | 0.333 |
| B12x, draw 1 | 0.989 | 0.957 / 0.929 | 0.667 | 0.417 |
| B12x, draw 2 | 0.989 | 0.986 / 0.986 | 0.333 | 0.333 |
| B12x `-exact`, draw 1 | 0.989 | 0.986 / 0.971 | 0.556 | 0.417 |
| B12x `-exact`, draw 2 | 0.989 | 1.000 / 0.986 | 0.667 | 0.417 |

Neither W4A4 nor the baked scales cost anything this eval resolves. Baked draw 1's v2 dip
did not repeat, alert's nine items move by 0.33 between draws, and both exact draws score
within the same spread [measured, two draws each]. **Verdict: not adopted.** B12x is the
faster kernel for one long cold prompt (21% at 98k), but on agents it costs more in KV
than it gains in compute.

### L: where the agent wall goes (p39, p40)

Row V left the agent wall unexplained below the prefill-compute floor. p39 traces it.

- **Method.**
  - `TRACE=1` starts the server with vLLM's torch profiler.
  - `probes/kstage/trace_agents.py` runs the 8-agent load and opens a 3 s profiler window
    at 15, 170 and 320 s after the first request. Those windows catch every agent at about
    12k, 73k and 108–112k tokens of context.
  - `probes/kstage/trace_steps.py` cuts each engine step at the sampler. It then cuts the
    step again at the residual-add RMSNorm that starts every layer, and charges each slice
    of wall, idle gaps included, to the layer type the slice holds.
  - Graph kernels run on several streams at once, so a sum of kernel times would overstate
    the wall.
- **The profiler costs wall.** The traced runs took 390 s and 389 s, against 351 s (p33)
  and 352 s (p37) for the same arms untraced [measured]. Flushing each window stalls the
  engine [inference]. The step splits are the evidence here, not the traced walls.

Steps by layer type. The rest of each step is input prep, lm_head, the sampler and copies.
Two arms were traced: prefix KV in RAM with 4,096-token chunks (`base-kvoff8-b4k`), and K6
with a 2 GiB cold tier, LFU and KV in RAM, with 8,192-token chunks (`live-lfu-cg2-kvoff8-b8k`,
KV 2.71 GiB, 382 experts cold, 16.6 a layer) [measured]:

| Arm | Context | Step | Wall | Attention | MoE | Mamba |
|---|---|---|---|---|---|---|
| prefix KV in RAM | ~75k | 4,096-token prefill | 673–707 ms | 53–54% | 36–38% | 8–9% |
| prefix KV in RAM | ~111k | 4,096-token prefill | 852–865 ms (4.7–4.8k tok/s) | 62–64% | 29–30% | 7% |
| prefix KV in RAM | ~13k | decode, 8 streams | 12.6 ms | 9% | 64% | 23% |
| K6 2 GiB | ~12k | decode, 8 streams | 14.6 ms | 7% | 71% | 19% |
| K6 2 GiB | ~73k | decode, 8 streams | 16.9 ms | 17% | | |
| K6 2 GiB | ~73k | 2,480-token prefill | 440 ms | 52% | | |
| K6 2 GiB | 108–112k | ~4,100-token prefill | 852–876 ms (4.6–4.9k tok/s) | 60–63% | 29–32% | |
| K6 2 GiB | 108–112k | decode, 7 streams | 17.4 ms | | | |

- **Deep prefill is attention-bound.** By kernel, the deepest 4,096-token step (865 ms)
  is [measured]:
  - FlashInfer's BatchPrefill: 533 ms (62%)
  - Marlin MoE: 182 ms (21%)
  - Marlin linear layers: 45 ms
  - nvjet GEMMs: 26 ms
  - LoRA expand: 10 ms
  - each Mamba kernel: about 5 ms
- **Attention grows with depth.** It adds about 4.75 ms for every 1k tokens of context,
  per 4,096-token chunk [arithmetic, from the two depths].
  - The work: six attention layers, 32 heads of 128 dims, 4,096 queries over 111k keys,
    is 44.7 TFLOP a step [arithmetic].
  - At 533 ms that is ~84 TFLOPS sustained [arithmetic]. The card's attainable peak for
    this kernel is unknown.
- **The cold tier costs decode, not the floor.**
  - At 12k, K6's decode step is 2 ms (16%) slower than no expert offload, and the extra time
    is in the MoE layers [measured].
  - At depth, both arms run the same ~860 ms attention-bound prefill step. Host-to-device
    copies take 202 ms of K6's 3.9 s deep window (~5%) [measured].
  - That is why 351 s and 352 s tie: the deep end of the load is bound by attention over
    the prefix, not by where the experts live [inference].
- **Why 8,192-token chunks do not help.** Each turn adds 4,096 + 256 = 4,352 new tokens
  [arithmetic], so its prefill never fills an 8k chunk. The deep K6 steps carry ~4.1k
  tokens either way [measured].
- **sm_120 has no fp8 prefill attention.** FlashInfer's fa2 prefill reads the fp8 KV,
  converts it to bf16 inside the kernel, and multiplies it against bf16 queries. TRTLLM
  prefill is unsupported on SM12x, which gets XQA decode only [sourced, vLLM 0.29.0
  `utils/flashinfer.py` `supports_trtllm_attention`]; the traces show XQA's `kernel_mha` on
  decode [measured]. `FLASH_ATTN` falls back to FA2 on sm_120, and takes fp8 KV only as
  FA3 on SM90 or FA4 on SM100, so it needs bf16 KV here [sourced, `fa_utils.py`
  `flash_attn_supports_kv_cache_dtype`]. `TRITON_ATTN` takes fp8 KV [sourced,
  `triton_attn.py`].
  p40 tries the levers this leaves.
- **`auto` KV is fp8 for this checkpoint.** `--kv-cache-dtype auto` resolves to the
  checkpoint's `kv_cache_quant_algo`, which is FP8 in its `hf_quant_config.json`
  [sourced, vLLM 0.29.0 `utils/torch_utils.py` `resolve_kv_cache_dtype_string`]. So p40's
  "kvbf16" arm logged "Using fp8_e4m3 data type to store kv cache" and ran as a second
  draw of the control (358 s vs 357 s), and its `FLASH_ATTN` arm died at startup on the
  same fp8 KV [measured]. Forcing bf16 takes `--kv-cache-dtype bfloat16`; p41 reruns both.
- **p40** [measured, one draw each; laptop, no expert offload, 8 GiB of KV in RAM, 4k
  chunks, 8 agents to 127k]:

  | Arm | GPU KV tokens | Wall | Decode median | Late TTFT | Preemptions |
  |---|---|---|---|---|---|
  | control (FlashInfer, fp8 KV) | 491,520 | 357 s | 35.5 tok/s | 8.45 s | 0 |
  | "kvbf16" (ran fp8, see above) | 478,412 | 358 s | 34.8 tok/s | 8.62 s | 0 |
  | `TRITON_ATTN`, fp8 KV | 609,484 | 417 s | 30.9 tok/s | 7.53 s | 1 |
  | `FLASH_ATTN`, "bf16" | — | did not start (fp8 KV) | | | |

  Triton's prefill is the faster one at depth (late TTFT −11%) and it holds 24% more
  tokens, but its decode is 13% slower, which sets the wall here. The padding arms are
  under "CUDA-graph limits, judged", item 1.
- **p41** [measured, one draw each; p40's load and settings with `--kv-cache-dtype
  bfloat16`]:

  | Arm | GPU KV tokens | Wall | Decode median | Late TTFT | Preemptions | Host RAM |
  |---|---|---|---|---|---|---|
  | bf16 KV, 8 GiB in RAM | 337,833 | 385 s | 45.6 tok/s | 11.81 s | 8 | +12.9 GiB |
  | bf16 KV, 16 GiB in RAM | 337,833 | 384 s | 45.6 tok/s | 11.80 s | 8 | +21.1 GiB |
  | bf16 KV, 16 GiB, `FLASH_ATTN` forced | 337,833 | 384 s | 45.5 tok/s | 11.79 s | 8 | +21.2 GiB |

  - With bf16 KV vLLM picks `FLASH_ATTN` by itself ("Using FLASH_ATTN attention backend out
    of potential backends: ['FLASH_ATTN', 'FLASHINFER', ...]") [measured]. Under fp8 KV
    `FLASH_ATTN` is not eligible, which is why the default has been FlashInfer. The forced
    arm is a second draw of the one above it.
  - Decode is 28% faster than the fp8 control (45.6 vs 35.5 tok/s), but late TTFT is 40%
    longer, eight requests were preempted, and the wall is 8% longer [measured]. The GPU
    holds 0.69 times the tokens: bf16 doubles the bytes per attention token, while each
    page's Mamba state stays the same size [arithmetic]. The lost room is the likely cost
    [inference].
  - 16 GiB of KV in RAM changes nothing against 8: both read the same 13,626,096 tokens
    back from RAM [measured], so 8 GiB of bf16 already holds what this load re-reads
    [inference].
  - Open: whether the faster decode is the kernel or the dtype, and whether bf16 KV wins
    once it has room. p42 forces FlashInfer on bf16 KV, and runs bf16 KV beside the K6
    expert cache, which frees GPU memory for it.
- **p42** [measured, one draw; p41's first arm with `--attention-backend FLASHINFER`]:
  276,912 GPU KV tokens, wall 381 s, decode median 56.9 tok/s, late TTFT 13.35 s, 25
  preemptions. The dtype makes the decode, not the kernel: FlashInfer on bf16 decodes 60%
  faster than on fp8 (35.5) and 25% faster than `FLASH_ATTN` on bf16 (45.6). It also
  leaves less room, 1.8 GiB of KV against `FLASH_ATTN`'s 2.2 at the same settings
  [measured], so more requests are preempted, and the GPU served 44,016 prefix tokens
  against the fp8 control's 609,696: almost every hit (13,705,744 tokens) was read back
  from RAM. The wall sits between the two (381 s vs 357 and 385). p42's K6 arm died on a
  parse bug in the arm name (`-kvbf16` read as a KV size) before it started; p43 reruns it.
- **p43** [measured, one draw each; p40's K6 arm (2 GiB cold tier, KV pinned at 2.7 GiB,
  8 GiB of KV in RAM, 8k chunks) with bf16 KV]:

  | Arm | GPU KV tokens | Wall | Decode median | Late TTFT | Preemptions |
  |---|---|---|---|---|---|
  | K6, fp8 KV (p40) | 737,280 | 354 s | 30.5 tok/s | 5.56 s | 7 |
  | K6, bf16 KV (`FLASH_ATTN`, picked by vLLM) | 415,369 | 389 s | 37.8 tok/s | 9.30 s | 41 |
  | K6, bf16 KV, FlashInfer | 415,369 | 360 s | 44.9 tok/s | 10.36 s | 14 |

  - FlashInfer beats `FLASH_ATTN` on bf16 again: decode +19%, a third of the preemptions,
    wall −7% [measured].
  - Against fp8, bf16 on FlashInfer decodes 47% faster in 0.56 times the tokens, its late
    TTFT is 86% longer, and its wall is 2% longer [measured]: inside the 351–359 s spread
    of every K6 arm so far, so a tie [inference].
  - K6's cold misses cost bf16 decode as they cost fp8: 44.9 tok/s against 56.9 without
    K6 (p42) [measured].
  - Open: whether bf16 wins once it holds as many tokens as fp8. p44 gives it a 3 and a
    4 GiB cold tier with KV left to vLLM.
- **p44** [measured, one draw each; p43's FlashInfer bf16 arm with a 3 and a 4 GiB cold
  tier and KV left to vLLM, against the fp8 arms with the nearest token counts]:

  | Arm | GPU KV tokens | Wall | Decode median | Late TTFT | Preemptions |
  |---|---|---|---|---|---|
  | K6 3 GiB, fp8 KV (p37 r2) | 1,009,254 | 359 s | 25.3 tok/s | 2.72 s | 0 |
  | K6 3 GiB, bf16 KV | 566,747 | 359 s | 32.8 tok/s | 6.39 s | 14 |
  | K6 2 GiB, fp8 KV (p37) | 740,556 | 352 s | 29.2 tok/s | 5.05 s | 5 |
  | K6 4 GiB, bf16 KV | 710,742 | 369 s | 27.5 tok/s | 4.61 s | 3 |

  - bf16 does not win once it has room. To hold fp8's 2 GiB tier's tokens it needs a 4 GiB
    tier, and the extra cold experts cost more decode than bf16 gains: 27.5 tok/s against
    29.2, wall +5% [measured]. At an equal 3 GiB tier it decodes 30% faster in 56% of the
    tokens and ties the wall (359 s) [measured].
  - bf16 KV is closed as a wall lever on the laptop [inference from p41–p44: no bf16 arm
    beat its fp8 twin by more than the 351–359 s spread]. It stays a decode lever: a
    caller that wants tokens per second per agent, not the whole load done sooner, gains
    28–60% from it.

### The agent load on the pool and on the desktop

The laptop is one of the lab's two cards. The same agent loads ran on the pool (both cards,
pipeline parallel 2 through gpu-lab's `lab pool up`: `probes/pool_agents.sh`) and on the
desktop's RTX 3090 alone (`probes/desktop_agents.sh`). Both were promised on 2026-10-07 and
ran on 2026-10-08.

- **Pool** [measured, one draw each; Lightning NVFP4 with G6q, CUDA graphs, 16 sequences,
  0.85 of each card, 4k chunks unless named, no KV in RAM]:

  | KV | GPU KV tokens | Agents | Wall | Decode median | Late TTFT | Preemptions |
  |---|---|---|---|---|---|---|
  | fp8 (FlashInfer) | 4,800,512 | 8 | 354 s | 34.0 tok/s | 6.13 s | 0 |
  | fp8 (FlashInfer) | 4,800,512 | 16 | 615 s | 14.8 tok/s | 5.57 s | 0 |
  | bf16 (`FLASH_ATTN`, picked by vLLM) | 2,813,432 | 8 | 341 s | 29.4 tok/s | 3.91 s | 0 |
  | bf16 (`FLASH_ATTN`, picked by vLLM) | 2,813,432 | 16 | 639 s | 12.7 tok/s | 3.79 s | 0 |
  | fp8, 8k chunks | 4,220,518 | 8 | 339 s | 39.2 tok/s | 6.55 s | 0 |
  | fp8, 8k chunks | 4,220,518 | 16 | 624 s | 16.6 tok/s | 7.61 s | 0 |
  | fp8, 512 chunks (`bin/lab`'s default) | 5,026,611 | 8 | 363 s | 23.5 tok/s | 2.03 s | 0 |
  | fp8, 512 chunks (`bin/lab`'s default) | 5,026,611 | 16 | 624 s | 11.9 tok/s | 2.09 s | 0 |

  - Against the laptop's best: 8 agents tie (351–357 s), 16 agents finish 12% sooner (697–
    698 s, p33–p37), and late TTFT is lower in every pool arm [measured].
  - The pool holds 4.8M fp8 tokens, more than twice the 2.03M that 16 agents at 127k need
    [arithmetic]. Nothing is read from RAM, and 91% of prompt tokens hit the prefix cache on
    the cards (27.4M of 30.0M at 16 agents) [measured].
  - bf16 on the pool cuts late TTFT 32–36% but decodes 14% slower, and the wall moves −4%
    at 8 agents and +4% at 16 [measured]. Single cards decode faster on bf16 (laptop and
    desktop both +28%); why the pool does not is unknown.
  - 8k chunks move the pool's wall −4% at 8 agents and +1% at 16, inside one draw's
    noise [inference], and cost 12% of its KV (4.22M tokens), twice what 16 agents need
    [measured]. Decode is 12–15% faster and late TTFT 7–37% longer, as on the laptop.
  - The lab's default 512-token chunks cost the wall 3% at 8 agents and 1% at 16, and cut
    late TTFT to a third (2.03–2.09 s against 5.57–6.13); decode falls 20–31% [measured].
    The default serves this load about as well as 4k [inference], with the lowest late
    TTFT of any setup here, on any node [measured]. On one card b512 was refuted (less
    KV, above); on the pool it holds the most KV of the three chunk sizes [measured].
- **Desktop** [measured, one draw each; RTX 3090 (sm_86), `--linear-backend marlin`, fp8 KV
  unless named, 8 agents to 127k unless named]:

  | Arm | GPU KV tokens | Wall | Decode median | Late TTFT | Preemptions |
  |---|---|---|---|---|---|
  | 8 GiB of KV in RAM, 4k chunks | 645,529 | 444 s | 27.5 tok/s | 8.42 s | 2 |
  | 8 GiB of KV in RAM, 8k chunks | 629,145 | 443 s | 28.5 tok/s | 9.19 s | 8 |
  | 16 GiB of KV in RAM, 4k chunks | 645,529 | 447 s | 27.2 tok/s | 8.48 s | 2 |
  | no KV in RAM, 4k chunks | 645,529 | over 2,400 s (212 of 224 turns) | | | |
  | no KV in RAM, 8k chunks (d41's "K6" arm: K6 never loaded) | 629,145 | over 2,400 s (213 of 224 turns) | | | |
  | bf16 KV (`FLASH_ATTN`), 8 GiB in RAM, 4k chunks | 374,755 | 480 s | 35.2 tok/s | 13.96 s | 4 |
  | K6 (4 GiB cold tier, every slot), no KV in RAM, 8k chunks | 1,658,060 | 739 s | 11.0 tok/s | 3.91 s | 0 |
  | 16 agents, 8 GiB of KV in RAM, 4k chunks | 645,529 | 884 s | 17.0 tok/s | 24.91 s | 0 |
  | 16 agents, 16 GiB of KV in RAM, 4k chunks | 645,529 | 880 s | 17.1 tok/s | 24.87 s | 0 |

  - The 3090 takes 24% longer than the laptop's card on this load (444 s vs 357) [measured].
  - KV in RAM is what lets the load finish here. Eight agents at 127k need 1.02M tokens
    [arithmetic] against 645k on the card, so evicted prefixes are computed again, and the
    bench used up its 2,400 s with 212 turns done. With 8 GiB in RAM, 11.1M tokens come back
    over the 12.2 GB/s link instead [measured].
  - 16 GiB of KV in RAM first could not start on the desktop. vLLM's native host KV lives
    in `/dev/shm`, which was half of RAM there: "Insufficient space in /dev/shm: 16382 MiB
    required, 15356 MiB free" [measured]. `/dev/shm` is now 24G in the desktop's
    `/etc/fstab` (2026-10-08, David's go).
  - With room for it, 16 GiB of KV in RAM changes nothing: 447 s against 444 at 8 agents,
    880 s against 884 at 16 (d44) [measured]. The tokens served from RAM are identical to
    the count with 8 GiB: 11,124,864 at 8 agents and 27,106,416 at 16 [measured]. 8 GiB
    already holds every prefix the load comes back for, so the wall here is compute and
    the 12.2 GB/s link, not RAM capacity [inference].
  - bf16 KV on the 3090 behaves as on the laptop: vLLM picks `FLASH_ATTN`, decode +28%,
    late TTFT +66%, wall +8% [measured].
  - d41's K6 arm ran without K6. The desktop copy lacked `vllm_kstage-0.1.dist-info`, the
    entry point that registers the plugin, and `PYTHONPATH` alone does not load it. Its
    host RAM rose 3.8 GiB, not K6's 15 [measured]. `desktop_agents.sh` now stops an arm
    whose plugin never logs "kstage: installed", and d43 reran it.
  - With K6 loaded (d43), the experts leave the card and KV gets 6.05 GiB: 1.66M tokens,
    more than eight agents need, so the load finishes without KV in RAM, with 0
    preemptions and 90% of prompt tokens hitting the card's prefix cache [measured]. It
    is 1.66× slower than KV in RAM (739 s vs 444) [arithmetic]: every cold miss crosses
    the 12.2 GB/s link, 2.79 misses a layer-step, and decode falls to 11.0 tok/s
    [measured]. On this link K6 buys room with 40% of the speed; KV in RAM, which moves
    blocks only when a prefix comes back rather than on every step, is the lever here
    [inference].
  - 16 agents with 8 GiB of KV in RAM finish in 884 s: 27% slower than the laptop's best
    (697 s) and 44% slower than the pool (615 s) [arithmetic], with late TTFT 24.91 s
    against the pool's 5.57 [measured]. K6 with KV in RAM (25 GiB pinned) does not fit
    beside the monitoring stack in 30 GiB.

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
     cheap [inference]. It is now steps J and K of "Experts in RAM": first the measured
     routing histogram, then a placement build.
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
| Hot/cold expert bits | expert offload traffic | untested | In the "Experts in RAM" program (J, K) |

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

## CUDA-graph limits, judged

An outside assessment (2026-10-08) proposed four CUDA-graph directions. Each was checked
against vLLM 0.29.0's source and the p39 traces; the padding claim was also run (p40).

1. **Padding to capture sizes ("9 → 16 is 44% wasted").**
   - Graphs serve only steps of 32 tokens or fewer: capture sizes 1, 2, 4, 8, 16, 24 and 32.
     The agent load's 4k prefill steps and mixed steps exceed that and run without a graph.
     Its deep decode steps run 3–8 streams [measured, p39].
   - Decode here is bound by memory reads: expert weights and KV. A padded row adds math to
     a step whose time is set by those reads, so 7 padded rows in 16 do not cost 44%
     [inference]. A search found no measurement behind the 44% [sourced, agent search].
   - **Lightning's router does pad into real experts** [sourced]:
     - Its config (`n_group` 1, `topk_group` 1, a score bias, scaling 2.5) keeps
       `GroupedTopKRouter`, because dropping the single group would change the advertised
       routing method from DeepSeekV3 to Unspecified (`router_factory.py`,
       `fused_moe/config.py` `get_routing_method_type`).
     - `VLLM_MOE_SKIP_PADDING` (default on) hands the padding mask to the fused routers
       only; `grouped_topk_router.py` never reads it.
     - The V2 runner leaves stale token ids in padded rows, so they route like tokens.
     - The result: expert weights read for nobody, and under K6, cold misses that copy
       experts nobody asked for [inference].
   - p40 test: `-cs8` captures every size from 1 to 8, so steps of 3, 5, 6 and 7 streams pad
     by no rows instead of 1–3. KV is pinned (`-kvb270`), because more capture sizes inflate
     vLLM's graph-memory estimate, which would otherwise move KV.
   - **Result** [measured, one draw each; K6 with a 2 GiB cold tier, 8 GiB of KV in RAM,
     8k chunks, 8 agents to 127k]:
     - wall: 351 s with every size captured, 354 s without;
     - decode median: 30.5 tok/s for both;
     - late TTFT: 5.51 s vs 5.56 s;
     - K6 misses: 0.83 vs 0.82 per layer-step (141,057 in 170,786 vs 139,650 in 170,053).
     Padding moves nothing past noise. A padded row's stale token likely routes to experts
     the real rows already pulled in, and graph steps are a small share of this load's wall
     [inference]. **Verdict: not a lever for the agent load.** At most it is a side test for
     decode-heavy serving, where 9–15 streams pad to 16; not a research effort.
2. **"Graphs take 1–2 GB of VRAM."**
   - The graph pool here is 0.07–0.10 GiB [measured, "The graph-memory estimate"].
   - What costs KV is vLLM's estimate of it. vLLM profiles the first capture and adds a
     second sample for each further size [sourced, agent read of `gpu_worker.py`]. It
     guesses 0.59–0.65 GiB for 0.08–0.09 GiB of real pool.
     `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` takes 0.60 GiB back (row 13).
   - The other cost is the compiled model's peak activation. `FULL_DECODE_ONLY` graphs
     without torch.compile give +0.87 GiB of KV (row T).
   - The "cg2" and "cg3" in our run names are K6's cold tier in GiB (`KSTAGE_COLD_GB`), not
     graphs.
   - **Verdict:** no new lever. The two real ones are measured, and adopting them is
     David's call.
3. **Conditional nodes (CUDA 12.3+) for speculative decode and expert skip.**
   - vLLM 0.29.0 uses no conditional nodes and no `cudaGraphExecUpdate` [sourced, grep of
     the image's vllm]. FlashInfer ships conditional nodes in `fused_moe/da_moe.py`, which
     vLLM does not import [sourced].
   - Marlin MoE already gives no work to an expert no token routed to: its block alignment
     makes blocks only for routed experts [sourced, `moe_align_block_size.py`].
   - Draft models did not pay on Lightning (DFlash, MTP). A conditional node would save a
     host round-trip; it would not raise the acceptance rate, which is what decides the
     gain [inference]. NVIDIA's description:
     https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes
   - **Verdict:** a separate later effort at most; it needs engine changes.
4. **Kernel fusion and megakernels.**
   - Hazy's "No Bubbles" is batch-1 Llama-1B on H100
     [sourced, https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles]. Mirage's MPK
     ran on A100, H100 and B200 at batch 1–16, including one MoE (Qwen3-30B-A3B), as a
     torch.compile backend, with no Mamba and no sm_120
     [sourced, https://arxiv.org/html/2512.22219v1].
   - What they remove is GPU idle time between kernels. Here that is 75 ms of a 3,023 ms
     decode window (2.5%) and about 1% of a deep prefill window [measured, p39]. Fusion
     could also save activation round-trips through memory; that is unmeasured [unknown].
   - **Verdict:** not a lever for this load.

What the assessment missed: at depth the agent wall is attention over the prefix, 54–64% of
each 4,096-token prefill step (row L, "L: where the agent wall goes"). No graph change
touches it.

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

No reference flag in `probes/` or `bin/lab` was changed. Each switch below changes a reference
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
- **Usage reports.** vLLM posts a usage report to stats.vllm.ai unless `DO_NOT_TRACK`,
  `VLLM_DO_NOT_TRACK` or `VLLM_NO_USAGE_STATS` is set (vLLM 0.29.0 `usage_lib.py`), and
  LMCache 0.5.4 posts to stats.lmcache.ai every 600 s unless `DO_NOT_TRACK` or
  `LMCACHE_TRACK_USAGE=false` is set [sourced]. The reports carry hardware, the model
  (vLLM: its architecture) and settings, with no host name or address field, though each
  request shows the server the network's public address [sourced].
  `probes/desktop_agents.sh` (from d51) and `probes/pr37190.sh` now pass
  `-e DO_NOT_TRACK=1`; `bin/lab`, `kv_levers.sh` and the eval servers do not, so their
  runs have likely reported [inference]. Adding it changes no measurement.
- **`bin/lab`'s Lightning serving** (gpu-lab repo): the same flags. At 0.92 vLLM leaves
  1.9 GiB of the card for everything else, and the desktop session and apps already take
  0.2–1.1 GiB of it. So close apps before starting, or get the display off the card
  [arithmetic; measured earlier for the apps].

## Reproduce

All on the laptop, with Chrome and Antigravity closed, from the worktree:

```
bash probes/kv_levers.sh round4    # rows 2-9 (one server per phase, evals into results/kv-levers/)
bash probes/kv_levers.sh round5    # rows 10-14
bash probes/kv_levers.sh round6    # Experts in RAM: A and C (offload_bench.py workloads)
bash probes/kv_levers.sh round8    # B and G
bash probes/kv_levers.sh round7    # D-J
python3 probes/kv_levers_agree.py > results/kv-levers/agree.txt
```

The CPU-only probes run in the serving image (`--network none --pull never`). The command
lines are in each probe's docstring:

- `probes/mamba_state_precision.py` reads the NVFP4 snapshot.
- `probes/expert_delta.py` reads the BF16 snapshot `a9904d2`.


Row N's llama.cpp arms, from the worktree. On the desktop, copy `probes/` to `~/moe-qlora-agents`
as for `probes/desktop_agents.sh`, build and convert there too, and set `BENCH_S=4800` (the
client's limit; the default is 2,400 s):

```
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp-src/llama.cpp
git -C ~/llama.cpp-src/llama.cpp checkout de7fa0a
bash probes/llamacpp/build.sh      # llama-server into ~/llama.cpp-src/llama.cpp/build/bin
S=/srv/model-cache/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/bee7596271d1495f6992ae224aefde4410e816b8
docker run --rm --pull never --entrypoint sh -v "$HOME/llama.cpp-src/llama.cpp":/src:ro \
  -v /srv/model-cache:/srv/model-cache:ro -v "$HOME/gguf":/out -e HOME=/tmp --user "$(id -u):$(id -g)" \
  vllm/vllm-openai:v0.29.0 -c "cd /tmp && PYTHONPATH=/src/gguf-py python3 /src/convert_hf_to_gguf.py \
  $S --no-mtp --outtype bf16 --outfile /out/lightning-nvfp4.gguf"     # 19,789,001,664 bytes
bash probes/llamacpp/serve.sh ll-gpu-a8l ll-cmoe-ub4096-t8-a8l ll-cmoe-mc12000-ub4096-t8-a8l
bash probes/llamacpp/serve.sh ll-cmoe-mc8500-ub4096-t8-a16l ll-cmoe-mc10000-ub2048-t8-a16l
bash probes/llamacpp/serve.sh ll-cmoe-mc13000-ub2048-t8-kvq4-a16l ll-ncm10-ub2048-t8-kvq4-a16l ll-ncm24-ub2048-t8-a16l
bash probes/llamacpp/serve.sh ll-gpu-ub2048-kvq4-a8l ll-gpu-ub1024-kvq4-a8l ll-cmoe-mc15000-ub2048-t8-kvq4-a8l
bash probes/llamacpp/serve.sh ll-cmoe-mc10000-ub2048-t8-kvq4-a16l ll-ncm16-mc2000-ub2048-t8-kvq4-a16l
bash probes/llamacpp/serve.sh ll-ncm10-ub2048-t16-kvq4-a16l ll-ncm10-ub2048-t24-kvq4-a16l ll-cmoe-mc13000-ub2048-t16-kvq4-a16l
bash probes/llamacpp/serve.sh ll-ncm8-ub1024-t16-kvq4-a16l ll-ncm10-ub2048-t16-omb1-kvq4-a16l
bash probes/llamacpp/serve.sh ll-ncm8-ub1024-t24-kvq4-a16l ll-ncm8-ub1024-t16-omb1-kvq4-a16l \
  ll-ncm8-ub1024-t16-omb1-np16-kvq4-a16l
bash probes/llamacpp/serve.sh ll-gpu-ub2048-np8-cram16384-kvq4-a16l ll-gpu-ub2048-np12-cram16384-kvq4-a16l \
  ll-gpu-ub2048-np10-cram16384-kvq4-a16l     # 16 agents on fewer slots, idle slots swapped through RAM
WORK=decode BARGS="--conc 1,16 --gen 512" bash probes/llamacpp/serve.sh ll-ncm10-ub2048-t16-omb1-kvq4-smoke \
  ll-ncm10-ub2048-t16-kvq4-smoke     # decode only, run under nvidia-smi dmon -s t
bash probes/llamacpp/serve.sh ll-gpu-ub2048-a8l ll-gpu-kvq4-a16l ll-gpu-ub256-kvq4-a16l \
  ll-cmoe-mc10000-ub4096-t8-a16l ll-ncm8-ub2048-t16-kvq4-a16l ll-ncm10-ub4096-t16-kvq4-a16l \
  ll-ncm6-ub1024-t16-kvq4-a16l ll-ncm6-ub512-t16-kvq4-a16l ll-ncm8-ub1536-t16-kvq4-a16l   # do not fit
bash probes/llamacpp/serve.sh ll-cmoe-mc13000-ub2048-t8-kvq4-np16-a16l ll-ncm10-ub2048-t8-kvq4-np16-a16l   # second draws
# desktop (BENCH_S=4800)
bash probes/llamacpp/serve.sh ll-cmoe-mc12000-ub4096-t8-a8l ll-cmoe-mc10000-ub2048-t8-a16l
bash probes/llamacpp/serve.sh ll-gpu-ub2048-kvq4-a8l ll-cmoe-mc15000-ub2048-t8-kvq4-a8l ll-cmoe-mc13000-ub2048-t8-kvq4-a16l
bash probes/llamacpp/serve.sh ll-ncm10-ub2048-t8-kvq4-a16l ll-ncm8-ub2048-t8-kvq4-a16l ll-ncm8-ub2048-t10-kvq4-a16l ll-ncm8-ub2048-t4-kvq4-a16l
bash probes/llamacpp/serve.sh ll-ncm10-ub4096-t8-kvq4-a16l ll-ncm8-ub3072-t8-kvq4-a16l
bash probes/llamacpp/serve.sh ll-ncm8-ub2560-t10-kvq4-a16l ll-ncm8-ub2048-t20-kvq4-a16l
bash probes/llamacpp/serve.sh ll-ncm8-ub2560-t10-omb1-kvq4-a16l ll-ncm8-ub2816-t10-kvq4-a16l
bash probes/llamacpp/serve.sh ll-gpu-ub2048-np8-cram16384-kvq4-a16l

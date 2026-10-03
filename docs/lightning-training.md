# Training Nemotron 3.5 Lightning on one 24 GB GPU: the regimen

The short version of `docs/plan.md`, as of 2026-10-02. Lightning (NVIDIA Nemotron 3.5
Lightning, 30B parameters with about 3B active per token, a hybrid of Mamba and attention
layers with 128 routed experts per MoE layer) trains with LoRA on the RTX 5090 Laptop by
holding its routed experts in 4-bit. The best adapter, **G6q**, beats the lab's previous
default (Qwen3-8B with its own research adapter) in both thinking modes and loses no row
to it. Every number below is MEASURED in `docs/plan.md` unless it says otherwise; the
plan has the files each one came from.

## Where things stand

| | |
|---|---|
| Candidate adapter | **G6q** (`results/research_dataset_g6q.json`, 1,248 records) |
| Backup adapter | **G6u**: G6q's data plus reasoning traces; better with thinking off on noticing asserted fake flags (trap3), worse on plain lookups with thinking on |
| Yardstick | `Qwen/Qwen3-8B` + the `qwen3-8b-research-v3` LoRA (r=16), the lab's previous default, on the same evals |
| Trained from | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, rev `a9904d24`, experts NF4 at load |
| Served on | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, rev `bee75962`, vLLM `v0.29.0` |
| Adapter weights | `~/moe-qlora/results/<name>-train-adapter/` on the laptop (not in git); served copies at `/srv/model-cache/adapters/lightning-{g6q,g6u}` (desktop cache, laptop mirror) |

## The recipe

Everything here is `probes/g6_train.py`, run by `probes/gpurun.sh` in gpu-lab's training
image on the laptop. The run goes through `probes/attn_bf16.py`, which keeps attention in bf16.

**Loading the model (Route 1, settled by gates G1-G5).**
- transformers' own `nemotron_h` class. Each fused expert stack (128 experts per layer)
  is quantized to NF4 the moment it reaches the GPU (bitsandbytes `replace_parameter_4bit`,
  adapted from axolotl). The other linears load as NF4 too.
- Attention q/k/v/o (24 projections in 6 layers) and `lm_head` stay bf16. With them in NF4,
  Route 1 failed the forward-fidelity gate (G2). Kept in bf16, it passed.
- The "lean scan" replaces the Mamba layers' memory-hungry torch scan (no mamba kernels
  are installed). Cross-entropy runs in 256-token chunks, so the 131,072-entry
  vocabulary's logits never sit in memory whole.
- No fp32 upcast of the non-4-bit parameters. Gradient checkpointing is on (non-reentrant)
  and the batch is 1. Peak memory is 19.4-19.9 GiB of the card's 23.4.

**What LoRA trains ("placement A").** Rank 16, alpha 32, dropout 0.05, no bias, on every
attention projection, every Mamba `in_proj` and the shared experts' up/down projections:

    .*\.mixer\.(q_proj|k_proj|v_proj|o_proj|in_proj)$|.*\.mixer\.shared_experts\.(up_proj|down_proj)$

That is 11.4M trainable parameters in 93 modules. The 128 routed experts are frozen.

**Optimisation.** Paged AdamW 8-bit, lr 1e-4, no weight decay, cosine schedule with 3%
warmup, gradient-norm clip 1.0, 2 epochs, 8 records per optimizer step (batch 1, loss
averaged over all assistant tokens in the step), seed 0, a fresh record order each epoch,
max length 2048 tokens, loss on assistant tokens only. These are the settings the Qwen3-8B
v3 adapter trained with (gpu-lab `training/qlora.py`), so the comparison is like for like.

**Chat rendering (`RENDER=think`, "design A").** Lightning's thinking-on prompt ends with
`<think>\n`, so for records in thinking mode that opener is masked as prompt and the model
learns to write `</think>` and then the answer. Thinking-off records get the empty
`<think></think>` the server sends, also masked. One sequence per record, so earlier turns
carry `<think>\n</think>` where the server sends `<think></think>`. Design B (one sequence
per turn) removes that newline and lost (G6ub below). A parity guard checks every
trained turn against the server's prompt and refuses to train if one differs.

**Thermals.** `GUARD=hw` aborts on hardware thermal or power-brake flags or at 90 C. The
card holding its own 87 C target is counted, not fatal. G6q ran at 77 C max and G6u at 79 C
and 153 W, with no abort.

**Cost.** G6q: 312 steps, 2,970 s (50 min). G6u: 338 steps, 3,338 s (56 min). The gate
evaluation takes another ~15 min.

**Reproduce G6q from scratch (laptop):**

    DATASET=/out/research_dataset_g6q.json RENDER=think GUARD=hw MAX_LEN=2048 \
      probes/gpurun.sh g6q-train /probes/attn_bf16.py g6_train.py g6q-train
    LABEL=g6q probes/g6_eval.sh results/g6q-train-adapter     # serve base + adapter, run the evals
    LABEL=g6q probes/g6_compare.py                             # the gate, against base

G6u is the same with `DATASET=/out/research_dataset_g6u.json RENDER=trace` and label `g6u`.
`DRY_RUN=1` renders and masks the data, prints samples and stops before loading the model.

## The data

Each version adds to the last. All datasets are committed under `results/`. The builders
are `probes/g6*_build.py`.

| dataset | records | what it adds |
|---|---|---|
| v3 (gpu-lab) | 950 | the Qwen3-8B v3 adapter's data: tool-backed flag lookups, traps (fake flags to deny) and no-tool questions; 693 thinking "default", 257 "off"; no reasoning text |
| G6p | 1,150 | 200 task-shaped records: a goal, answered with the whole procedure rather than one flag (`g6p_build.py`) |
| **G6q** | **1,248** | 98 tool-choice records: 58 that answer live questions with the `promql` tool against the lab's Prometheus (real tool outputs at build time) and 40 that write Prometheus alert rules directly, each passing `promtool` (`g6q_build.py`); checked against the eval for contamination |
| G6t | 1,248 | 131 records whose thinking-on turns are base Lightning's own reasoning, kept only where the eval's own scorer accepts the answer (`g6t_collect.py`, `g6t_build.py`) |
| G6u | 1,248 | traces on 403 records: 131 from base, 272 from G6t (12 whole, 260 partial) |
| G6w | 1,248 | G6u with its 60 web-research records showing `web_search` failing (`g6w_build.py`); dropped |

## How an adapter is judged

**The evals.** gpu-lab's research eval through `probes/g7a_eval.py` (patched for Lightning's
XML tool calls and its open `<think>`), at temperature 0:

| set | what it measures |
|---|---|
| v1 | flag lookups on held-out tools, seen tools, traps, no-tool questions |
| v2 | held-out flags, two flags in one question, fix-the-command, tasks |
| rocky | Rocky Linux tools and tasks, with traps |
| promql / promqlcat | live Prometheus questions (promqlcat adds the metric catalog) |
| general | plain questions that need no tool |
| alert | writing alert rules, checked with promtool |
| trap3 | does it notice an asserted fake flag, or make one up |

The gate runs all seven in both modes: thinking on at 4,096 tokens (primary) and thinking
off at 512. The repeat studies (N1, L7, L8) use five of them with thinking on.

**The gate (`probes/g6_compare.py`).** Against base Lightning in the same mode, the adapter
must win held_out and lose none of trap, trap_control, no_tool, task, rocky_task, alert or
promql, where a win or loss means more than 4 items apart. Thinking on is primary.

**The yardstick and the noise bar.** Repeat runs of the same server differ by a few items
(N1: thinking on, N2: thinking off), and 13 different serving setups put base v1 held_out
anywhere from 42 to 53 of 90. So since 2026-10-02 a row counts as a win or loss only at a
two-sided sign test p < 0.05 on the items where the two sides differ, with three repeats a
side where possible, and with the pooled count and the Holm-corrected count reported
beside it (P1, `probes/pair_items.py`). A single run at p < 0.05 is a lead, not a result.

## Version history

| version | change | gate (on / off) | outcome |
|---|---|---|---|
| G6 | v3 as is, 1,024 tokens | fail / fail | task rows lost: the model answers goals with one flag (template collapse); ties Qwen v3 (1 W, 0 L) |
| G6r | design-A thinking render | fail / fail | fewer turns stuck in an unclosed `<think>` (64 to 24); task rows unchanged |
| G6p | + 200 task records | fail / **pass** | task rows recover; promql gets worse (bash lookups instead of the promql tool) |
| G6p2048 | max length 2,048 | fail / pass | same as G6p |
| **G6q** | + 98 tool-choice records | **pass / pass** | first to pass thinking on; promql 0.833 and alert 7/9 thinking on; vs Qwen v3 off: 2 W, 0 L |
| G6t | + base reasoning traces | pass / pass | reasons on 447 of 970 thinking-on turns (G6q: 0 of 919); task wins vs base |
| G6u | + more traces | pass / pass | task 0.95 and trap3 noticed 0.75 thinking on; was on `main` from fa7a944 |
| G6v | trap traces | not trained | 0 of 424 attempts accepted: it looks up the tool, then double-checks with web_search |
| G6w | web_search-failure data | pass / pass | dropped: fabricates more on trap3, trails on promql; web_search on traps only 72 to 62 of 123 |
| G6ub | G6u with design-B render | not a gate run | lost alert thinking on 0-8 (p 0.008): it fell back on bash help lookups; design A stays |

**Choosing the candidate (L7, L8; item level, three repeats a side):**

| | thinking off vs Qwen v3 | thinking on vs Qwen v3 | head to head |
|---|---|---|---|
| G6u | 2 W, 0 L | 3 W, **3 L** | off: +1 (trap3) |
| G6q | 1 W, 0 L | 1 W, 0 L | on: +2 (held_out2, rocky_trap) |

G6q is the only adapter that beats the yardstick in both modes without losing a row.
G6u's reasoning wins the task-shaped rows and loses plain lookups. Use **G6q** by default.
Keep **G6u** for a thinking-off workload where noticing fake flags matters most, or where
readable reasoning is wanted: G6u reasons on 555 of 994 thinking-on turns, G6q on none, so
G6q's "thinking on" is thinking off with a longer budget.

## Serving the adapter

All on vLLM `v0.29.0` with the NVFP4 checkpoint. Serving on NVFP4 costs nothing measurable
against FP8 (L6), though training ran on bf16 + NF4. Always serve with CUDA graphs:
`--enforce-eager` cost 6.7-8x decode speed and changed no answer.

| where | flags beyond the base set | decode, 1 user / 16 users (tok/s per request) | KV room |
|---|---|---|---|
| laptop gate server (`g6_eval.sh`) | eager, util 0.85: the gate's reference, not for use | ~33 (base Lightning, eager) | 1.5 GiB (base) |
| desktop 3090 alone | `--linear-backend marlin --kv-cache-dtype bfloat16`, graphs, util 0.95 | 179.6 / 92.4 (G6u) | 2.0 GiB |
| pool (both cards) | `G7D_ADAPTER=g6q G7D_EAGER= G7D_MORE="--kv-cache-dtype bfloat16" probes/g7d_pool.sh` | 176.5 / 75.0 (G6u) | 1.53M tokens |

The base set is `--kv-cache-dtype fp8 --mamba-cache-mode align --moe-backend marlin
--max-model-len 16384 --max-num-seqs 16 --enable-lora --max-lora-rank 16 --max-loras 1
--lora-modules g6q=/hf/adapters/lightning-g6q`.

- The 3090 needs `--linear-backend marlin`, or vLLM picks a CUTLASS FP8 kernel sm_86
  cannot run.
- A LoRA adapter costs 16-17% of decode speed.
- G6q and G6u share every flag and the same adapter size, so G6u's speed figures carry
  over (INFERENCE; G6q has not been benchmarked).

**Speculative decoding does not combine with the adapter here.**
- DSpark (+70% for base Lightning on one 3090) does not fit beside a LoRA adapter on one
  3090: KV is 0.19 GiB short at util 0.95.
- No draft model runs pooled in vLLM 0.29.0: DSpark and DFlash are refused, and MTP hangs.

## Levers that are closed

- **More data of the same kinds.** G6w and G6v show the trace and web-search levers are
  spent, and design B lost.
- **Routed-expert LoRA.** The remaining failures are behaviours that attention + shared
  experts already reach, and per-expert rank 16 does not fit one card.
- **Training across both cards (48 GB).** About a week to build, with no named ceiling to
  justify it.
- **Sampling above temperature 0.** Ranking would need many repeats per point.
- **FlashInfer's Mamba kernel on sm_86.** No gain under CUDA graphs. On sm_120 it hits a
  CUDA TMA bug (`probes/ssu_sm120/`). The upstream draft stays local.

## Open

- Whether Lightning joins the lab's front door, and on which node (David's call).
- G6q's own speed on the desktop and the pool (expected equal to G6u's; not measured).
- Pooled speculative decoding: recheck on the next vLLM image.

# Plan: the next base model after Lightning

Lightning is closed (plan.md, "Ledger close-out", 2026-10-02): G6q is the candidate,
G6u the backup. The open question is which base the next round trains.

- Smaller Nemotron. David's case: if tuning could bring it up to Lightning, or even to
  Qwen3.8, its smaller size would leave more room for context and cache.
- Qwen3.8-27B. It scores 34 on the Artificial Analysis index (xhigh effort), against 13 for
  Lightning and 7 for the Nano 4B and 9B v2. It needs about 19 GB at 4-bit, and its KV
  cache costs about 10× Lightning's per token.

The KV figures are ARITHMETIC from the pinned configs, fp8 KV, attention layers only:

| model | attention layers × KV heads × head dim | KV per token |
|---|---|---|
| Lightning | 6 × 2 × 128 | 3 KiB (about 9.2 KiB effective in vLLM, MEASURED) |
| Nano 4B | 4 × 8 × 128 | 8 KiB |
| Nano 9B v2 | 4 × 8 × 128 | 8 KiB |
| Qwen3.8-27B | 16 × 4 × 256 | 32 KiB |

David, 2026-10-03:

- "then download both models and run the eval sets"
- "feel free to go straight into training nano 4B on g6q if that's where the data takes you"
- "feel free to test both versions of nano 4B as well as testing 9B in general"

## S1: base screen, pre-registered 2026-10-03

### Models

All four are pinned, and each manifest is in `results/s1-<tag>-manifest.json` (HF tree API).

| tag | repo @ rev | size | precision |
|---|---|---|---|
| `q38-int4` | `RedHatAI/Qwen3.8-27B-INT4` @ `91bd022d` | 18.14 GiB | W4A16 (AWQ + GPTQ, group 128); vision tower, embeddings, lm_head and `linear_attn.in_proj_{a,b}` in bf16 |
| `nano4b-bf16` | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` @ `dfaf35de` | 7.42 GiB | bf16 |
| `nano4b-fp8` | `nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8` @ `3fe6dab7` | 4.92 GiB | modelopt FP8, FP8 KV |
| `nano9b-bf16` | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` @ `6533e8de` | 16.57 GiB | bf16 |

Why this Qwen3.8 build:
- It is the smallest 4-bit build from a known quantizer.
- W4A16 runs on Marlin on both sm_86 and sm_120.

The alternatives lost:
- `nvidia/Qwen3.8-27B-NVFP4` is mixed NVFP4/FP8 at 20.4 GiB, which leaves about 2 GiB
  less room on a 24 GB card.
- `Qwen/Qwen3.8-27B-FP8` is 28.7 GiB and needs the pool.

Downloads follow G0's method: on the desktop, as uid 1000, with `hf download --revision`
inside `gpu-lab:training`. Each is checked with `probes/g0_verify.py` against its manifest.

### Protocol

This is G6's eval, unchanged:
- `probes/g7a_eval.py`, gated by its `--selfcheck`, on all seven sets.
- Thinking on at 4096 tokens is the primary mode; thinking off at 512 is secondary.
- Flags: `--max-calls 3 --temperature 0 --window 4000 --seed 20260923 --concurrency 16`,
  plus `--promql-catalog` on promqlcat.

There is no adapter. The runner is `probes/s1_screen.sh`, one vLLM `v0.29.0` server per
model, with `--max-model-len 16384 --max-num-seqs 16`. It runs with CUDA graphs: ledger
L2 found graphs change no quality row. It also records:
- the KV capacity vLLM reports;
- bench.py decode rates at c=1 and c=16.

Those two answer the room question with measurements.

Template compatibility was CHECKED from each pinned `chat_template.jinja`:
- **Qwen3.8 and Nano 4B** ask for Lightning's XML tool-call form, read `enable_thinking`,
  and open `<think>\n` with thinking on. The harness runs them unchanged.
- **Nano 9B v2** ignores `enable_thinking`; a `/think` or `/no_think` tag in a system or
  user turn switches it. Its tool calls are `<TOOLCALL>[{"name":…, "arguments":…}]`,
  which the harness's JSON form reads. It runs with g7a_eval.py's new `--think-tag`.
  The selfcheck covers both the tag and the `<TOOLCALL>` form.
- **Qwen3.8's effort setting:** its template defaults to `reasoning_effort` xhigh, which
  adds a "think carefully" system line. So Qwen3.8's thinking-on pass runs twice: at the
  default (label `q38-int4`), and at low (`q38-int4-low`, via `--chat-kwargs
  '{"reasoning_effort":"low"}'`). Thinking off carries no effort line.

Per-model serving:
- **Qwen3.8** runs on the desktop 3090 with `--language-model-only` (the vision tower is
  not loaded) and `--kv-cache-dtype fp8` (Red Hat's calibrated KV scales). It uses
  `--gpu-memory-utilization 0.90`: the card drives no display, about 17 GiB of 4-bit
  weights leave little KV at 0.85, and 0.9 is the single-card setting David approved on
  2026-10-02.
- **The Nano builds** run on the laptop with vLLM's defaults for KV dtype, plus each model
  card's `--mamba-ssm-cache-dtype float32`. The parser flags are left out, because the
  harness reads raw text.

### Comparison

`probes/s1_compare.py` runs each model against three references:
- base Lightning, thinking on: the `{set}-lightning-think-4k` runs, with v1 from G6's
  server;
- base Lightning, thinking off: `{set}-lightning-nothink`;
- the G6q adapter, both modes.

Two rules apply:
- **Rows:** G6's row rule. A difference counts as a win or a loss only when it is more
  than 4 items on that split.
- **Pooled:** item-level sign test on the discordant items (`pair_items.py`'s method).

### Decision rules, CHOSEN before the run

- **Nano 4B training (David's go above).**
  - Train Nano 4B on G6q's data if base Nano 4B, thinking on, is within reach of base
    Lightning on the five rows no adapter here has been shown to fix: task, rocky_task,
    alert, promql, and general (correct_where_scorable). Within reach means it loses at
    most 2 of those 5.
  - Flag and trap rows don't count against it: the Qwen3-8B adapters took held-out flag
    lookups from 69% to 99% (memory: research-eval-8b-results).
  - The better of the two builds is the one trained, because training runs on the bf16
    weights either way.
- **Qwen3.8 and Nano 9B v2** are descriptive. They feed David's choice of the next base.
  No training follows from S1 alone.

### S1 result: Nano 4B BF16 (MEASURED, 2026-10-03)

`python3 probes/s1_compare.py nano4b-bf16`, laptop, 478 items per mode.

- **Room and speed at the screen's settings (0.85, bf16 KV, fp32 mamba state):**
  - 10.66 GiB of KV holds 403,950 tokens. Lightning holds 266,240 on the same card.
  - Decode runs 96.3 tok/s at c=1 and 70.8 per stream at c=16, against Lightning's ~220 at
    c=1. A dense 4B in bf16 reads 7.5 GiB per token; Lightning's ~3B active parameters
    are 4-bit.
- **Thinking on, against base Lightning:**
  - Rows: win 1, loss 6, tie 15. Pooled items: +44 for Nano 4B against +139 for Lightning,
    sign p < 0.001.
  - 59 items ended truncated in think (Lightning: 0).
  - The losses are the flag rows (held_out −35, seen_tool −11, held_out2 −18,
    rocky_held_out −13), trap −5 and rocky_task −10.
- **Thinking off, against base Lightning:** win 0, loss 11, tie 11.
- **Training rule: TRAIN.** Of the 5 reasoning rows, Nano 4B lost 1 (rocky_task, 0.15 vs
  0.65). It tied task (−3), alert (−1), promql (0) and general (0).
  - Caveat: the rule is lenient on small sets. The alert set has n=9, so a loss needs 5 of
    the 9 items.

### N4: Nano 4B on G6q's data, pre-registered 2026-10-03

The training run is G6q's recipe (docs/lightning-training.md) with `BASE=nano4b`.

- **Load:** bf16, whole. There is no 4-bit and no attn_bf16 wrapper.
- **Targets:** attention q/k/v/o, mamba in_proj, and the dense MLP up/down_proj.
- **The rest is G6q's:** data, render, length, LoRA rank and optimisation.
- **The DRY_RUN parity guard passes.** It counts the same 1,178,797 tokens and 2,424/2,424
  turn boundaries as Lightning's G6q render: the two tokenizers and templates render this
  data identically.

```
DATASET=/out/research_dataset_g6q.json RENDER=think GUARD=hw MAX_LEN=2048 BASE=nano4b \
  probes/gpurun.sh nano4b-g6q-train /probes/g6_train.py nano4b-g6q-train
```

Eval: `s1_screen.sh` with `ADAPTER=results/nano4b-g6q-train-adapter` on the BF16 base,
TAG `nano4b-g6q`, the same seven sets in both modes.

Readings, CHOSEN before the run:

1. **Did training help?** Compare N4 with base Nano 4B by G6's rule: it must win held_out
   and lose none of trap, trap_control, no_tool, task, rocky_task, alert and promql.
2. **David's question: does it reach Lightning?** Compare N4 with Lightning G6q, thinking
   on. N4 reaches it if it loses no row and the pooled sign test does not favour G6q at
   p < 0.05. The same reading is reported against base Lightning.
3. **Thinking off:** also reported against the Qwen3-8B v3 adapter, like for like on data.

### N4 result (MEASURED, 2026-10-03)

**Training.**
- The run took 312 steps in 2,320 s; G6q took 2,970 s. Loss went from 1.659 to 0.057
  (G6q: 1.713 to 0.074).
- Peak torch memory was 10.35 GiB, the card peaked at 81 °C, and there was no abort.
- The adapter has 17.05M trainable parameters in 71 modules (G6q: 11.36M in 93). Its
  files are `results/nano4b-g6q-train-adapter/`, which is gitignored like every adapter.

**Serving** on the BF16 base with LoRA, at the screen's settings:
- 426,548 tokens of KV;
- decode 91.4 tok/s at c=1 and 68.7 per stream at c=16.

`python3 probes/s1_compare.py nano4b-g6q`. Thinking on averages 75 completion tokens per
item, and no item was truncated in think.

**Reading 1, did training help? PASS in both modes** (G6 rule against base Nano 4B):
- Thinking on: win 9, loss 0. Pooled items +204 for N4, +16 for base.
- Thinking off: win 13, loss 0. Pooled items +287 for N4, +9 for base.

**Reading 2, does it reach Lightning? NO**, by the rule chosen before the run.
- **Against G6q, thinking on:**
  - Rows: win 0, loss 1, tie 21. Pooled items: +11 for N4 against +28 for G6q,
    sign p 0.0095.
  - The flag and trap rows are level (held_out 0.989 vs 0.989; trap2 0.917 vs 0.917).
  - Every gap is on a reasoning row:

    | row | N4 | G6q | difference |
    |---|---|---|---|
    | rocky_task | 0.15 | 0.45 | −6, the one LOSS |
    | task | 0.50 | 0.70 | −4 |
    | trap3 noticed | 0.08 | 0.42 | −4 |
    | alert | 0.67 | 0.78 | −1 |
    | promql | 0.78 | 0.83 | −1 |

- **Against G6q, thinking off:** win 0, loss 1 (trap3 noticed), tie 21. Pooled items +8
  against +29, sign p 0.0008.
- **Against base Lightning, thinking on:**
  - Win 9, loss 1, tie 12. Pooled items +121 against +28.
  - The adapted 4B far outscores the untrained 30B, but it still loses rocky_task (0.15 vs
    0.65), so it does not "reach" by the rule.

**Reading 3, thinking off, against the Qwen3-8B v3 adapter: level.** Win 2, loss 0, tie
20. Pooled items +21 against +19, sign p 0.87.

**What it means.**
- This bears out the prediction made before the run: the adapter carries the lab's
  habits to a 4B as completely as to Lightning, with flag rows at 99–100%.
- Rows that need the base model's own reasoning stay where the base left them. Nano 4B
  was 0.15 on rocky_task before training and 0.15 after.
- On this eval, a tuned Nano 4B is a smaller G6q with weaker task rows. Measured on the
  laptop at util 0.85:
  - decode is 91 tok/s at c=1 with the adapter, against base Lightning's ~220;
  - KV room is 1.6× base Lightning's 266,240 tokens.

### S1 result: Qwen3.8-27B INT4 (MEASURED, 2026-10-03)

`python3 probes/s1_compare.py q38-int4 q38-int4-low`, desktop 3090, 478 items per pass.

- **Room and speed at util 0.90, fp8 KV, language model only:**
  - Weights 16.84 GiB.
  - KV 1.63 GiB, which is **31,804 tokens**: about an eighth of Lightning's 266,240 on one
    card, and 1.94 requests of 16,384.
  - Decode 46.7 tok/s at c=1 and 38.1 per stream at c=16.
- **Thinking on, template default (xhigh):**
  - Against base Lightning: win 1, loss 4, tie 17. Pooled items +56 for Qwen3.8 against
    +77 for Lightning, sign p 0.08, so level.
  - 29 items ended truncated in think, at 806 completion tokens per item.
  - It calls tools heavily: on v1, 46 of 158 items reached the 3-call limit, and the
    harness records no answer for those.
  - Losses: held_out −12, seen_tool −6, trap −5, held_out2 −5.
  - Alert reached 1.00 against 0.56, but at +4 that is a tie by the row rule.
- **Thinking on, reasoning_effort low:**
  - Against base Lightning: **win 5, loss 0, tie 17. Pooled items +89 against +47,
    sign p 0.0004.**
  - Only 2 items were truncated, at 469 tokens per item.
  - Wins: trap +6, no_tool over_trigger +5, task (0.95 vs 0.60) +7, trap2 +5, general
    over_trigger +8.
  - Ties include alert (1.00 vs 0.56), rocky_task (0.65 vs 0.65), promql (0.78 vs 0.67)
    and trap3 noticed (0.58 vs 0.25).
- **Against G6q, thinking on, low:** win 1 (task 0.95 vs 0.70), loss 4, tie 17. Pooled
  items +24 against +92.
  - The four losses are all flag rows: held_out, seen_tool, held_out2 and rocky_held_out.
    Those are the rows an adapter fixes; N4 took Nano 4B's to 0.99–1.00.
- **Thinking off, against base Lightning:** win 6, loss 1 (task 0.40 vs 0.80), tie 15.
  Pooled items +130 against +51.
- **Effort is a measured confound.** Low effort beats xhigh on this protocol. xhigh
  spends its budget on tool loops that hit the 3-call cap and on longer thinking:

  | across all 478 items | xhigh | low |
  |---|---|---|
  | call_limit | 140 | 70 |
  | truncated_in_think | 29 | 2 |
  | items with no answer | 169 | 72 |
  | tool calls per item | 1.99 | 1.62 |
  - Reading: any Qwen3.8 work here should serve at low effort, or move the cap.

### S1 result: Nano 9B v2 BF16 (MEASURED, 2026-10-03)

`python3 probes/s1_compare.py nano9b-bf16`, laptop, with `--think-tag`.

The tag is checked on the live server:
- `/no_think` ends the prompt in `<think></think>`;
- `/think` and no tag both end it in `<think>\n`, so without the tag every "thinking off"
  run would have thought.

- **Room and speed:**
  - The weights are 16.58 GiB, which leaves 0.89 GiB of KV: **26,699 tokens**.
  - Decode runs 45.0 tok/s at c=1 and 42.8 per stream at c=16.
  - Tool calls arrive as `<TOOLCALL>` JSON, read by the harness as `json_embedded`.
- **Thinking on, against base Lightning:**
  - Rows: win 4, loss 7, tie 11. Pooled items: +77 for Nano 9B against +130 for
    Lightning, sign p 0.0003.
  - It wins every trap row: trap 0.93 vs 0.40, trap2, rocky_trap, and trap3 noticed
    (0.75 vs 0.25).
  - It loses the flag rows, plus rocky_task (0.35 vs 0.65) and general over_trigger.
- **Thinking off, against base Lightning:** win 2, loss 9, tie 11. It also denies the
  trap controls, which should not be refused: trap2_control 0.50 vs 0.08 and
  rocky_trap_control 0.64 vs 0.07. Part of its trap strength is a general readiness to
  refuse.

### S1 summary (MEASURED)

Every row is one card at the screen's settings: the laptop at util 0.85, except Qwen3.8,
which ran on the desktop 3090 at 0.90. Both "against base Lightning" columns give rows as
win/loss/tie, then pooled items better as model/Lightning.

| model | weights | KV tokens | decode c=1 | thinking on, vs base Lightning | thinking off, vs base Lightning |
|---|---|---|---|---|---|
| Lightning NVFP4 (reference) | 17.81 GiB | 266,240 | ~220 (v0.29: 216.6–224.6; 3090: 208.6) | | |
| Nano 4B BF16 | 7.47 GiB | 403,950 | 96.3 | 1/6/15; 44/139 | 0/11/11; 22/137 |
| Nano 4B FP8 | 5.01 GiB | 660,041 | 137.3 | 1/8/13; 44/153 | 1/11/10; 27/133 |
| Nano 9B v2 BF16 | 16.58 GiB | 26,699 | 45.0 | 4/7/11; 77/130 | 2/9/11; 59/120 |
| Qwen3.8-27B INT4, xhigh (3090) | 16.84 GiB | 31,804 | 46.7 | 1/4/17; 56/77 | 6/1/15; 130/51 |
| Qwen3.8-27B INT4, low (3090) | same | same | same | **5/0/17; 89/47** | (thinking off has no effort) |
| N4: Nano 4B + G6q adapter | 7.53 GiB | 426,548 | 91.4 | 9/1/12; 121/28 | 9/2/11; 190/27 |

### What S1 says about the next base (INFERENCE from the rows above)

- **David's room argument holds for Nano 4B and only for Nano 4B.**
  - It gives 1.5× Lightning's KV in BF16 and 2.5× in FP8.
  - Nano 9B v2 and Qwen3.8 both leave about a tenth of Lightning's room on one card.
  - None of the three small models decodes faster than Lightning: a dense model reads all
    of its weights for every token.
- **Training closes the habit gap, not the reasoning gap.**
  - N4 brings Nano 4B level with G6q on every flag and trap row.
  - It stays behind on the task, trap3 and alert rows, and G6q keeps the pooled lead.
- **Qwen3.8 is the stronger base.**
  - At low effort it already beats base Lightning with no training, and it loses to G6q
    only on the flag rows an adapter fixes.
  - Its cost is room: one 3090 holds 31,804 tokens, 1.94 requests of 16,384.
- **Two open questions for a Qwen3.8 round:**
  1. **Room on the pool: UNKNOWN.** Each stage would hold half the weights and half of the
     16 attention layers. Pooled PP for this architecture has not been run.
  2. **Training on one 24 GB card: UNKNOWN.** The nearest measurement is Qwen3-32B, with
     the same 64 layers and 5120 width. It fitted QLoRA at 1,024 tokens per step with
     0.1 GiB to spare, and only with `--no-fp32-upcast` (memory: laptop-training-ceilings).
     G6q's records reach 2,048 tokens, the vocabulary is 248k, and the Gated DeltaNet
     layers need training kernels the image may not have. A G4/G5-style memory probe
     comes first.

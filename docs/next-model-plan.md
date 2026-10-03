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
  - Decode runs 96.3 tok/s at c=1 and 70.8 per stream at c=16, against Lightning's ~225 at
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
  - decode is 91 tok/s at c=1 with the adapter, against base Lightning's ~225;
  - KV room is 1.6× base Lightning's 266,240 tokens.

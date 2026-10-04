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

**Correction, after the run: P1.** S1 pre-registered G6's 4-item rule a day after P1
(plan.md, adopted 2026-10-02) had replaced it for every comparison. So each result below
also gives the rows under P1:
- a row is a win or a loss only at a two-sided sign test p < 0.05 on its discordant
  items;
- the Holm count is beside it, over the 20 rows that have per-item scores
  (`correct_where_scorable` has none);
- a single run at p < 0.05 is a lead, not a result. "Repeats" below tests each model
  against each of N1's three runs of base Lightning and G6q.

`s1_compare.py` prints both rules, and writes both to `results/s1-compare.json`.

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
  - Under P1: win 0, loss 5, Holm 4 of 20. trap (0 items to 5, p 0.06) becomes a tie;
    the four flag rows and rocky_task stay losses.
- **Thinking off, against base Lightning:** win 0, loss 11, tie 11. Under P1: win 0,
  loss 10, Holm 3.
- **Training rule: TRAIN.** Of the 5 reasoning rows, Nano 4B lost 1 (rocky_task, 0.15 vs
  0.65). It tied task (−3), alert (−1), promql (0) and general (0).
  - Under P1 the same: it loses only rocky_task (0 items to 10, p 0.002). Nano 4B FP8,
    which lost 2 by the 4-item rule (task, rocky_task), loses none by P1.
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
- Under P1 it passes too: thinking on win 7, loss 0, Holm 7 of 20; thinking off win 10,
  loss 0, Holm 9. held_out is a win in both (53 items to 0; 73 to 1).

**Reading 2, does it reach Lightning? NO**, by the rule chosen before the run, and by P1.
- **Under P1, no row against G6q is a win or a loss, in either mode** (Holm 0 of 20).
  The pre-registered rule has a second half, the pooled test, and that is what says NO
  now: +11 against +28 thinking on (p 0.0095), +8 against +29 off (p 0.0008). It holds
  against each of G6q's three N1 runs ("Repeats" below).
- **Against G6q, thinking on:**
  - Rows: win 0, loss 1, tie 21. Pooled items: +11 for N4 against +28 for G6q,
    sign p 0.0095.
  - The flag and trap rows are level (held_out 0.989 vs 0.989; trap2 0.917 vs 0.917).
  - Every gap is on a reasoning row:

    | row | N4 | G6q | difference |
    |---|---|---|---|
    | rocky_task | 0.15 | 0.45 | −6, the one LOSS (items 1 to 7, p 0.07: a tie by P1) |
    | task | 0.50 | 0.70 | −4 |
    | trap3 noticed | 0.08 | 0.42 | −4 |
    | alert | 0.67 | 0.78 | −1 |
    | promql | 0.78 | 0.83 | −1 |

- **Against G6q, thinking off:** win 0, loss 1 (trap3 noticed, items 1 to 6, p 0.125: a
  tie by P1), tie 21. Pooled items +8 against +29, sign p 0.0008.
- **Against base Lightning, thinking on:**
  - Win 9, loss 1, tie 12. Pooled items +121 against +28.
  - Under P1: win 8, loss 1 (rocky_task, 1 item to 11, p 0.006), Holm 2 of 20.
  - The adapted 4B far outscores the untrained 30B, but it still loses rocky_task (0.15 vs
    0.65), so it does not "reach" by the rule.

**Reading 3, thinking off, against the Qwen3-8B v3 adapter: level.** Win 2, loss 0, tie
20 (P1: win 0, loss 0). Pooled items +21 against +19, sign p 0.87.

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
  - Under P1: win 1 (general over_trigger, 8 items to 0), loss 0, Holm 0 of 20. None of
    the four losses holds item by item: held_out 13 to 25 (p 0.07), seen_tool 2 to 8,
    trap 0 to 5 (p 0.06), held_out2 6 to 11.
  - Against base Lightning's three N1 runs it is level with one and below two
    ("Repeats").
- **Thinking on, reasoning_effort low:**
  - Against base Lightning: **win 5, loss 0, tie 17. Pooled items +89 against +47,
    sign p 0.0004.**
  - Only 2 items were truncated, at 469 tokens per item.
  - Wins: trap +6, no_tool over_trigger +5, task (0.95 vs 0.60) +7, trap2 +5, general
    over_trigger +8.
  - Ties include alert (1.00 vs 0.56), rocky_task (0.65 vs 0.65), promql (0.78 vs 0.67)
    and trap3 noticed (0.58 vs 0.25).
  - **Under P1: win 2, loss 0, Holm 0 of 20.** The wins are task (7 items to 0, p 0.016)
    and general over_trigger (8 to 0, p 0.008). trap (7 to 1), no_tool over_trigger and
    trap2 (5 to 0 each, p 0.06) become ties.
  - **Correction: "beats base Lightning" does not survive N1's repeats.** On the five
    contested sets, against each of base's three runs: 52/26 (p 0.004), 52/41 (p 0.30),
    48/35 (p 0.19). The S1 result leaned on base's low draw: base's r1, the run S1 used,
    is the lowest of its three (22 items to 37 against r2, 20 to 33 against r3; p 0.07,
    0.10). Restated: **at least level with base Lightning**.
  - Two more Qwen3.8 runs ("Repeats") keep it there. With three runs a side it leads on
    task (8 items to 1, p 0.039) and rocky_trap (8 to 0, p 0.008), loses no row, and no
    row survives Holm (0 of 14).
- **Against G6q, thinking on, low:** win 1 (task 0.95 vs 0.70), loss 4, tie 17. Pooled
  items +24 against +92.
  - The four losses are all flag rows: held_out, seen_tool, held_out2 and rocky_held_out.
    Those are the rows an adapter fixes; N4 took Nano 4B's to 0.99–1.00.
  - Under P1: win 0 (task is 6 to 1, p 0.125), loss 4 (the same four, each 0 or 1 items
    to 10–20), Holm 4 of 20.
- **Thinking off, against base Lightning:** win 6, loss 1 (task 0.40 vs 0.80), tie 15.
  Pooled items +130 against +51.
  - Under P1: win 4 (held_out, seen_tool, held_out2, rocky_held_out), loss 1 (task, 1 item
    to 9, p 0.021), Holm 3 of 20. two_flag and promql become ties.
- **Effort is a measured confound.** Low effort beats xhigh on this protocol: item by
  item, low against xhigh on all seven sets is 95 to 32 (p < 0.0001, one run each). xhigh
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
  - Under P1: win 4 (the same four trap rows), loss 3 (held_out, held_out2,
    rocky_held_out), Holm 1 of 20. seen_tool, fix_cmd, rocky_task (1 item to 7, p 0.07)
    and general over_trigger (4 to 11, p 0.12) become ties.
- **Thinking off, against base Lightning:** win 2, loss 9, tie 11. It also denies the
  trap controls, which should not be refused: trap2_control 0.50 vs 0.08 and
  rocky_trap_control 0.64 vs 0.07. Part of its trap strength is a general readiness to
  refuse.
  - Under P1: win 2, loss 7, Holm 0 of 20. rocky_trap_control stays a loss (0 items to 8,
    p 0.008); trap2_control (1 to 6, p 0.125) and task become ties.

### Repeats (MEASURED, 2026-10-03)

`python3 probes/s1_compare.py`; `results/s1-compare.json`, key `n1` and each model's `n1`.
Unless stated: thinking on, N1's five contested sets (v2, rocky, promqlcat, alert, trap3),
pooled items better as first/second, sign p.

**The null: each reference against its own repeats.**
- Base Lightning: 22/37 (p 0.07), 20/33 (p 0.10), 28/26 (p 0.89), so 53–59 discordant
  items. r1, the run S1 compared against, is the low one.
- G6q: 9/5, 6/5, 4/7: 11–14 discordant, p 0.42–1.0.
- G6q thinking off, all seven sets (L7): 12/6, 6/8, 3/11: 14–18 discordant, p 0.06–0.79.

**Nanos: every reading holds against each of the three runs.**
- Against base Lightning, each Nano base is below on every run: Nano 4B BF16 19/73, 15/84,
  14/81; FP8 25/84, 20/94, 21/93; Nano 9B v2 49/74 (p 0.03), 48/88, 42/80 (p < 0.001).
  N4 is above on every run: 70/27, 58/30, 57/27 (p ≤ 0.004).
- **N4 against G6q is robust.** Against each G6q run, thinking on: 9/27 (p 0.004), 12/26
  (p 0.03), 9/26 (p 0.006). That is 35–38 discordant items, all favouring G6q, against
  the null's 11–14. Thinking off, all seven sets: 8/29 (p 0.0008), 11/26 (p 0.02), 8/31
  (p 0.0003), which is 37–39 against 14–18.

**Qwen3.8 at low effort, two more runs.** Same flags and harness as S1, through
`s1_screen.sh REP=N PASSES=think-low`, both at util 0.90:
- r2 on the desktop 3090;
- r3 on the laptop's RTX 5090 Laptop, the first Qwen3.8 serve on that card.

All 14 evals exit 0. Truncated in think: 0 and 1 items (S1: 2). Completion tokens per
item: 443 and 459 (S1: 469).
- **Against itself:** 13/15, 9/17, 11/17, so 26–28 discordant, p 0.17–0.85. On all seven
  sets: 24/26, 13/25, 18/28 (p 0.07–0.89). That is between G6q's null and base's.
- **Against base Lightning's S1 references, all seven sets:** r2 wins 6 and loses 0 by
  the 4-item rule, P1 2/0, Holm 0, pooled 89/45. r3: 6/0, P1 3/0, Holm 0, pooled 101/47.
- **Each Qwen3.8 run against each base run:**

  | Qwen3.8 low | base r1 | base r2 | base r3 |
  |---|---|---|---|
  | S1 (3090) | 52/26 (p 0.004) | 52/41 (p 0.30) | 48/35 (p 0.19) |
  | r2 (3090) | 53/25 (p 0.002) | 50/37 (p 0.20) | 47/32 (p 0.11) |
  | r3 (laptop) | 60/26 (p 0.0003) | 54/35 (p 0.06) | 54/33 (p 0.03) |

  All nine pairs lean Qwen3.8. Four reach p < 0.05, and three of those four are against
  base's low r1.
- **Three runs a side** (P1, with each item's mean over a side's runs, as `n2_items.py`):
  - Against base Lightning: win 2 (task 8 items to 1, p 0.039; rocky_trap 8 to 0,
    p 0.008), loss 0, Holm 0 of 14. Pooled 83/45.
  - Against G6q: win 1 (alert 6 to 0, p 0.031), loss 2 (held_out2 1 to 19,
    rocky_held_out 1 to 23), Holm 2 of 14. Pooled 34/73.
- **Against G6q, each run:** below on every pairing, 23–29 items against 50–60
  (p 0.0001–0.014). On all seven sets under P1, each of the three runs loses exactly the
  four flag rows (held_out, seen_tool, held_out2, rocky_held_out) and nothing else.
- **xhigh, for comparison:** against base's three runs, 27/39 (p 0.18), 26/53 (p 0.003),
  27/52 (p 0.007). Level with r1 and below r2 and r3.
- **Reading:** "beats base Lightning" was one draw. At low effort Qwen3.8 is at least
  level with base Lightning on every row, and ahead on task at three runs a side. Its
  losses to G6q stay the flag rows.

**Reasoning rows in items.** Thinking on. Three-run columns give the mean (range); single
runs are plain.

| row (n) | Qwen3.8 low ×3 | base Lightning N1 | G6q N1 | Qwen3.8 xhigh | N4 | Nano 9B v2 |
|---|---|---|---|---|---|---|
| task (20) | 17.67 (17–19) | 13.00 (12–14) | 14.67 (14–15) | 12 | 10 | 15 |
| rocky_task (20) | 13.33 (13–14) | 12.67 (12–13) | 8.00 (7–9) | 10 | 3 | 7 |
| promql (18) | 13.67 (13–14) | 12.00 (12–12) | 13.67 (13–15) | 15 | 14 | 13 |
| alert (9) | 9.00 (9–9) | 6.00 (5–7) | 5.00 (3–7) | 9 | 6 | 3 |
| trap3 noticed (12) | 7.33 (7–8) | 3.67 (3–4) | 5.00 (5–5) | 1 | 1 | 9 |

Against base, item by item at three runs a side:
- task is the one reasoning row that reaches p < 0.05 (8 to 1).
- alert (5 to 0, p 0.06) and trap3 noticed (6 to 1, p 0.125) lean the same way but do
  not reach it.
- rocky_task (8 to 4) and promql (4 to 2) are level.

**Laptop against desktop for Qwen3.8**, both at util 0.90 with fp8 KV, language model only:

| card | weights | KV | KV tokens | requests of 16,384 | decode c=1 | c=16, per stream | 7 sets, wall time |
|---|---|---|---|---|---|---|---|
| desktop 3090 (r2) | 16.84 GiB | 1.63 GiB | 31,804 | 1.94 | 46.8 | 38.1 | 1,558 s |
| laptop RTX 5090 Laptop (r3) | 16.84 GiB | 0.99 GiB | 19,275 | 1.18 | 41.0 | 36.9 | 2,194 s |

- r2 reproduces S1's desktop serve: 31,804 tokens, and 46.8 tok/s at c=1 against 46.7.
- The laptop runs about half as many requests at once. Its serve log shows 3.6 running
  on average, with 11–13 waiting; the desktop's shows 6.7 running, with 8–9 waiting.
  Neither card holds the harness's 16 at once.
- Why the laptop's KV is smaller is not separately measured. Two things the 3090 lacks
  were measured:
  - the card reports 24,463 MiB against the 3090's 24,576;
  - during the run, nvidia-smi showed desktop apps holding about 0.6 GiB of it
    (gnome-shell 285 MiB plus five smaller clients).

  vLLM charges other clients' memory to KV (memory: vllm-kv-charges-other-gpu-clients).
  ARITHMETIC: the two together come to about 0.70 GiB, close to the 0.64 GiB gap.

### S1 summary (MEASURED)

Every row is one card at the screen's settings: the laptop at util 0.85, except Qwen3.8,
at 0.90 on the desktop 3090, and on the laptop for its r3 repeat. Both "against base
Lightning" columns give three readings against base's S1 reference runs:
- rows as win/loss/tie by the 4-item rule;
- P1 as win/loss, with the Holm count of 20;
- pooled items better as model/Lightning.

"Repeats" tests each model against base's other two runs.

| model | weights | KV tokens | decode c=1 | thinking on, vs base Lightning | thinking off, vs base Lightning |
|---|---|---|---|---|---|
| Lightning NVFP4 (reference) | 17.81 GiB | 266,240 | ~220 (v0.29: 216.6–224.6; 3090: 208.6) | | |
| Nano 4B BF16 | 7.47 GiB | 403,950 | 96.3 | 1/6/15; P1 0/5, Holm 4; 44/139 | 0/11/11; P1 0/10, Holm 3; 22/137 |
| Nano 4B FP8 | 5.01 GiB | 660,041 | 137.3 | 1/8/13; P1 0/4, Holm 3; 44/153 | 1/11/10; P1 0/9, Holm 3; 27/133 |
| Nano 9B v2 BF16 | 16.58 GiB | 26,699 | 45.0 | 4/7/11; P1 4/3, Holm 1; 77/130 | 2/9/11; P1 2/7, Holm 0; 59/120 |
| Qwen3.8-27B INT4, xhigh (3090) | 16.84 GiB | 31,804 | 46.7 | 1/4/17; P1 1/0, Holm 0; 56/77 | 6/1/15; P1 4/1, Holm 3; 130/51 |
| Qwen3.8-27B INT4, low (3090) | same | same | same | 5/0/17; P1 2/0, Holm 0; 89/47 | (thinking off has no effort) |
| Qwen3.8 low, repeat r2 (3090) | same | 31,804 | 46.8 | 6/0/16; P1 2/0, Holm 0; 89/45 | |
| Qwen3.8 low, repeat r3 (5090 Laptop) | same | 19,275 | 41.0 | 6/0/16; P1 3/0, Holm 0; 101/47 | |
| N4: Nano 4B + G6q adapter | 7.53 GiB | 426,548 | 91.4 | 9/1/12; P1 8/1, Holm 2; 121/28 | 9/2/11; P1 9/1, Holm 5; 190/27 |

### What S1 says about the next base (INFERENCE from the rows above)

- **David's room argument holds for Nano 4B and only for Nano 4B.**
  - It gives 1.5× Lightning's KV in BF16 and 2.5× in FP8.
  - Nano 9B v2 and Qwen3.8 both leave about a tenth of Lightning's room on one card.
    Qwen3.8 on the laptop, with the display's clients on the card, gets 19,275 tokens,
    about 7%.
  - None of the three small models decodes faster than Lightning: a dense model reads all
    of its weights for every token.
- **Training closes the habit gap, not the reasoning gap.**
  - N4 brings Nano 4B level with G6q on every flag and trap row.
  - It stays behind on the task, trap3 and alert rows, and G6q keeps the pooled lead. No
    single row is a loss by P1. The pooled lead holds against each of G6q's three runs, in
    both modes.
- **Qwen3.8 is the stronger base.**
  - At low effort, with no training, it is at least level with base Lightning (corrected
    from "beats"; see "Repeats"). With three runs a side it leads on task and rocky_trap
    and loses no row; no row survives Holm.
  - In all three of its runs it loses to G6q only on the four flag rows an adapter fixes.
  - task is the one reasoning row where it is ahead by P1: 17.67 of 20 over three runs,
    against base's 13.00 and G6q's 14.67.
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

## Q1: does Qwen3.8-27B QLoRA-train on one 24 GB card? Pre-registered 2026-10-03

This answers open question 2 above, by measurement on the laptop (RTX 5090 Laptop, sm_120,
24,463 MiB = 23.89 GiB), the card G5 and G6 trained on. Nothing below is run yet.

### Weights

- `Qwen/Qwen3.8-27B` @ `1d4bf0f2`, BF16: 32 files, 55,586,114,863 B (51.77 GiB), 19 LFS
  (18 shards and tokenizer.json). The manifest is `results/s1-q38-bf16-manifest.json`
  (SOURCED, HF tree API). The script that wrote it reproduces
  `s1-qwen38-int4-manifest.json` byte for byte from that repo's tree.
- Downloaded on the desktop as uid 1000 with `hf download --revision` in
  `gpu-lab:training` (container `dl-q38-bf16`), as S1's were.
- **Pass (CHOSEN):** `probes/g0_verify.py` passes on the desktop snapshot and again on the
  laptop mirror after `lab mirror sync` (`results/s1-q38-bf16-verify-{desktop,laptop}.json`).
  The probe reads the mirror.

### The probe: `probes/q1_probe.py`, through `probes/gpurun.sh`, `GUARD=hw`

**Load (G4's style).**
- `Qwen3_5ForCausalLM`, text only, built from the checkpoint's `text_config`. SOURCED in
  the image: transformers 5.16.1 maps `model.language_model.*` onto it
  (`conversion_mapping.py:978`, a `PrefixChange` for `qwen3_5_text`) and ignores
  `^model.visual.*` and `^mtp.*` (`modeling_qwen3_5.py:1584`). So the vision tower and the
  MTP head are never built. Missing and unexpected keys are recorded, beside the index's
  key counts by prefix.
- NF4, double quantization, bf16 compute: qlora.py's load. The embeddings and the untied
  lm_head stay bf16 (bitsandbytes' default skip of the output layer); nothing is fp32.
- ARITHMETIC for the resident weights: 24.35B decoder Linear parameters at NF4 with
  statistics ≈ 11.70 GiB; embeddings and lm_head 2 × 1.271B × 2 B = 4.74 GiB; ≈ 16.44 GiB
  in all. A result more than 5% off is a surprise to explain, not a fail.
- Recorded: load time, host peak RSS, torch peak and idle allocated and reserved, NVML
  before, peak and idle, resident bytes by class, and a sanity forward (finite, with
  " Paris" top-1 for "The capital of France is"). Then the same idle figures with LoRA
  attached.

**Training steps (G5's style).**
- **LoRA:** r 16, alpha 32, dropout 0.05, bias none (G6q). Targets are G6q's regex
  carried over:
  - full-attention q/k/v/o_proj (16 layers);
  - the Gated DeltaNet input projections `in_proj_qkv`, `in_proj_z`, `in_proj_b` and
    `in_proj_a` (48 layers). Lightning's mamba `in_proj` fuses the same streams (x, z,
    B/C, dt);
  - MLP up/down_proj (64 layers).

  Not `gate_proj` or the DeltaNet `out_proj`: G6q trained no counterpart (Nemotron's MLP
  has no gate, and G6q did not target mamba `out_proj`). ARITHMETIC: 384 LoRA modules,
  85.0M trainable parameters. The lab's all-linear Qwen surface (plus `gate_proj` and
  `out_proj`) would be 116.7M; that one is not measured here.
- Gradient checkpointing (non-reentrant), `enable_input_require_grads`, no fp32 upcast
  (qlora.py `--no-fp32-upcast`, G5, G6q).
- `paged_adamw_8bit`, lr 1e-4 held constant, weight decay 0, clip 1.0. Batch 1, one
  sequence per optimizer step. G6q accumulates 8 records per step; the memory per
  sequence and per optimizer step is the same.
- **Chunked cross-entropy, chunk 256** (G6q's F1). `chunked_ce.chunked_loss` is reused
  under a `Qwen3_5ForCausalLM` forward written in the probe; the nemotron_h install is
  not used. Why: at 2,048 × 248,320 the bf16 logits are 0.95 GiB and fp32 1.89 GiB
  (ARITHMETIC), before the loss's own copies.
- **Kernel path.** MEASURED in the image: `fla`, `causal_conv1d`, `flash_attn` and
  `kernels` do not import. So the 48 DeltaNet layers run transformers' torch fallback,
  `torch_chunk_gated_delta_rule` (`modeling_qwen3_5.py:249`), with the torch conv1d;
  attention runs sdpa. The probe counts calls to the function actually used. Nothing is
  installed or patched: an OOM inside the fallback is a result.
- **Data.**
  - G6q's 1,248 records (`results/research_dataset_g6q.json`), rendered with Qwen3.8's
    template and the lab's TOOLS plus each record's `extra_tools`.
  - G6q's think render: thinking-on records mask `<think>\n`, the end of the thinking-on
    prompt, and train `\n</think>\n\n` + content + `<|im_end|>`. "off" records mask
    `<think>\n\n</think>\n\n`. Thinking-on records render at `reasoning_effort` low
    (S1's reading).
  - A boundary guard, as g6_train.py's `parity()`: on every trained turn, the span from
    the assistant header to the first trained token must equal the server prompt's.
  - The records are concatenated in a seeded order (seed 0) and cut into exact L-token
    windows with G6q's labels. Each step gets a new window.
- **Lengths:** 512, 1,024, 2,048, in that order. Per length, 2 warm-up and 10 measured
  steps.
- **Per step:** loss, grad norm, seconds, tokens/s, torch peak allocated and reserved,
  NVML peak, and the device peak by G5's definition (the larger of the sampled NVML peak
  and NVML-before − torch-reserved-before + the step's torch peak reserved). Per length:
  peak temperature, peak power and throttle-reason counts.
- **Thermal guard:** `GUARD=hw`, as the lab's training runs: abort on hw_thermal or
  hw_power_brake, or at 90 C; sw_thermal is counted. gpurun.sh sets `max-power` and
  restores `performance` on exit.

**Token counts** (a dry run, no GPU): every record's Qwen3.8 token count, untruncated, at
low and at xhigh effort: p50, p90, p95, max, and the number over 512, 1,024 and 2,048.
Lightning's are MEASURED (`results/g6q-train.json`): 1,178,797 tokens, p50 860, p90 1,503,
max 1,892. Qwen3.8's are UNKNOWN.

### Pass rule, CHOSEN before the run

- **A length PASSES** if all 12 steps finish with finite losses, no OOM and no guard
  abort, and the device peak stays ≤ card total − 0.5 GiB = 23.39 GiB (G5's line).
- A length that finishes but peaks over the line is "runs, over the line": not a pass,
  as Lightning's seq 2048 was before chunked CE.
- **An OOM ends the sweep.** It is a hard stop: no retry and no changed setting.

What each outcome means:
- **2,048 passes:** Qwen3.8 trains on one card on G6q's data as it is, if no record is over
  2,048 Qwen tokens. If some are, they would be cut as MAX_LEN cuts them, and the count
  is stated.
- **1,024 passes, 2,048 does not:** one card trains only records of ≤ 1,024 Qwen tokens
  (the number over is stated), or split ones; otherwise the pool.
- **Only 512 passes:** the same at 512. G6q's data would need rewriting short, so in
  practice the pool.
- **512 fails:** not trainable on one card in this configuration. The pool is the route
  (plan.md line 17's fallback), and whether it fits is UNKNOWN.
- Speed is recorded, not gated: epoch hours for G6q's Qwen token count at the measured
  tokens/s (ARITHMETIC).

### Q1 result (MEASURED, 2026-10-03): 2,048 PASSES, so Qwen3.8 trains on one card on G6q's data as it is

**Weights.** `dl-q38-bf16` exited 0. `g0_verify.py` PASSES on both snapshots: 32/32 files,
55,586,114,863 B, 19 sha256 and 13 git-blob hashes equal, all world-readable
(`results/s1-q38-bf16-verify-desktop.json`, then `lab mirror sync`, 55.59 GB in 211 s,
then `results/s1-q38-bf16-verify-laptop.json`).

**Token counts** (`results/q1-dry.{json,log}`; CPU only, `DRY_RUN=1`). Qwen3.8's tokenizer
and template render G6q's data almost exactly as Lightning's do:

| render | tokens | p50 | p90 | p95 | max | over 512 | over 1,024 | over 2,048 |
|---|---|---|---|---|---|---|---|---|
| Lightning (g6q-train.json) | 1,178,797 | 860 | 1,503 | | 1,892 | | | 0 |
| Qwen3.8, effort low | 1,190,982 | 862 | 1,500 | 1,543 | 1,901 | 1,102 | 491 | 0 |
| Qwen3.8, effort xhigh | 1,201,674 | 870 | 1,510 | 1,546 | 1,913 | 1,116 | 492 | 0 |

- 105,943 trained (assistant) tokens. Every record has some; 357 are "off" records.
- The boundary guard PASSES: 2,424/2,424 trained turns match the server prompt, the
  same count as G6q's render. 785 turns differ in history only: the training sequence
  tokenizes a past turn's `<think>\n\n</think>` as `<think>`, `\n`, `\n`, `</think>`,
  where the server writes `\n\n`. This is design A's history difference.

**Load** (`results/q1-laptop.{json,log}`, `GUARD=hw` via gpurun.sh, `max-power`):
- `Qwen3_5ForCausalLM` loaded in 11.3 s from the laptop mirror. The index holds 850
  `model.language_model.*` keys, 333 `model.visual.*`, 15 `mtp.*` and `lm_head`; the load
  reports 0 missing, 0 unexpected and 0 mismatched keys. 0 visual and 0 MTP modules were
  built; every parameter is on cuda:0.
- 496 `Linear4bit` modules; the only bf16 Linear left is `lm_head`. Attention runs sdpa.
- **Idle torch-allocated 16.441 GiB, against 16.44 arithmetic.** That is 11.339 NF4 packed,
  4.741 bf16 and 0.361 of quantization statistics. Torch peak during the load was 16.567;
  NVML peak 18.239; host peak RSS 38.9 GiB.
- Sanity forward: finite, " Paris" top-1.
- With LoRA attached: 85,008,384 trainable fp32 parameters in 384 modules, as the
  arithmetic said, and no other trainable parameter. Idle 16.789 GiB allocated, NVML
  18.378.
- Other apps held 1.222 GiB of NVML memory before CUDA started (nautilus, chrome and
  ptyxis on the display). The device peaks below include it.

**Kernel path.** `torch_chunk_gated_delta_rule` and `causal_conv1d_fn` are transformers'
own functions (`modeling_qwen3_5`); `fla`, `causal_conv1d`, `flash_attn` and `kernels` do
not import. Each training step calls the DeltaNet fallback 96 times (48 layers, forward
plus checkpoint recompute) and the chunked CE once. Nothing was installed or patched.

**Training steps.** 12 of 12 at every length, finite losses, no OOM, no abort. Throttle
reasons: sw_power_cap only (the card at its power limit; counted, not fatal). No thermal
or power-brake flag.

| seq | verdict | torch peak alloc | torch peak reserved | device peak (line 23.39) | s/step | tok/s | loss first → last | peak temp, power |
|---|---|---|---|---|---|---|---|---|
| 512 | **PASS** | 17.938 GiB | 18.416 | 20.091 | 2.115 | 242.1 | 1.453 → 0.736 | 74 °C, 166 W |
| 1,024 | **PASS** | 18.790 | 19.105 | 20.788 | 4.145 | 247.0 | 0.447 → 1.341 | 78 °C, 176 W |
| 2,048 | **PASS** | 20.628 | 21.051 | 22.708 | 9.737 | 210.3 | 0.517 → 1.324 | 78 °C, 173 W |

- The torch peak is the same on every measured step at each length, so it does not grow.
- Each step is a new window of different data, so first and last loss show only that the
  losses are finite. They are not a learning curve.
- The margin at 2,048 is 0.68 GiB under the line, with 1.2 GiB of other apps' memory
  inside the peak.

**Verdict under the rule chosen before the run: 2,048 PASSES, and no G6q record is over
2,048 Qwen tokens (max 1,901 at low effort).** So Qwen3.8-27B QLoRA-trains on one 24 GB
card on G6q's data as it is.

What it does not settle:
- **Speed (ARITHMETIC).** At 247 tok/s for ~1,000-token sequences and 210 at 2,048, one
  epoch of G6q's 1,190,982 tokens takes 1.3–1.6 h, so G6q's 2 epochs take about 2.7–3.1 h.
  G6q on Lightning took 0.83 h (2,970 s).
- **The LoRA surface.** The G6q-mapped 85.0M parameters were measured. The lab's
  all-linear Qwen surface (116.7M) is not. ARITHMETIC: its 31.7M extra fp32 parameters,
  gradients and 8-bit Adam states come to about 0.3 GiB, plus their activations, against
  a 0.68 GiB margin. UNKNOWN until measured.
- **Stability without the fp32 upcast** over a full run is UNKNOWN. The memory note on the
  32B says the same.
- **Pool room and fidelity** are not touched here. Whether the NF4 model keeps the base's
  eval rows is a G2-style question this probe does not ask.

## Q2: Qwen3.8-27B on G6q's data, pre-registered 2026-10-03

David, 2026-10-03: "proceed with the Qwen training run on the laptop", after Q1 passed.

### The run: `probes/q2_train.py`, through `probes/gpurun.sh`, `GUARD=hw`

- **Load, render, targets:** Q1's, unchanged: `Qwen/Qwen3.8-27B` @ 1d4bf0f2, NF4 on load,
  text only, no fp32 upcast, gradient checkpointing, chunked cross-entropy (chunk 256), LoRA
  r16 on Q1's `TARGETS` (attention q/k/v/o, DeltaNet in_proj_qkv/z/b/a, MLP up/down). Thinking-on
  records render at `reasoning_effort` low, the effort Qwen3.8 is served at.
- **Optimisation: G6q's**, as `g6_train.py` runs it: 2 epochs, 8 records per optimizer step at
  batch 1, the loss normalised over every assistant token in the step, paged AdamW 8-bit,
  lr 1e-4, weight decay 0, cosine schedule with warmup round(3% of steps), clip 1.0, seed 0,
  a fresh record order per epoch from `torch.Generator(seed + epoch)`. One sequence per
  record, cut at 2,048 tokens: none is longer (Q1: max 1,901). 1,248 records make 312 steps.
- **Before the run:** a `DRY_RUN` that must reproduce Q1's token counts and 2,424/2,424
  boundary parity, then a 2-step smoke adapter that vLLM v0.29.0 must load on the INT4 base
  with every trained module applied (no "ignored" LoRA modules), answering one request.
- **Hard stops:** an OOM or a guard abort ends the run; the adapter so far is saved as
  `-partial`. Moving any setting after a stop needs David.

```
DATASET=/out/research_dataset_g6q.json GUARD=hw MAX_LEN=2048 \
  probes/gpurun.sh q2-train /probes/q2_train.py q2-train
```

### Eval

`s1_screen.sh` on the desktop 3090 with `ADAPTER=results/q2-train-adapter` on the INT4 base
(`RedHatAI/Qwen3.8-27B-INT4` @ 91bd022d, util 0.90, fp8 KV, language model only), TAG
`q38-g6q`, `PASSES="think-low nothink"`: the seven sets thinking on at low effort and
thinking off, the S1 protocol otherwise.

**Amended before any Q2 eval ran: the desktop, not the laptop.** The smoke adapter's load
test on the laptop at these flags failed: 0.65 GiB of KV room against the 0.81 GiB one
request of 16,384 tokens needs ("estimated maximum model length is 9408",
`results/q2-smoke-vllm.log`). A shorter `--max-model-len` would change the protocol: the
thinking-on pass allows 4,096 new tokens after prompts of up to 6,099, about 10.2k. The
same smoke adapter on the desktop at the same flags, 2026-10-03:

- **Room:** 1.15 GiB of KV, 23,130 tokens, 1.41 requests of 16,384; the base alone had 1.63
  GiB and 31,804 there (S1, r2).
- **Applied:** at temperature 0 the adapter's token logprobs differ from the base's (mean
  0.0095, max 0.064 over 44 tokens), while base against base and adapter against adapter
  are identical. vLLM's DEBUG log skips only modules the adapter does not train (48
  `out_proj`, 48 `conv1d`, `embed_tokens`, `lm_head`).

Two of base Qwen3.8's three low-effort runs (S1, r2) were served on this card as well.

**A known confound:** the adapter trains against NF4-quantized BF16 weights and is served on
the W4A16 INT4 build, the only Qwen3.8 that fits one card for serving. Lightning's G6q has
the same kind of gap (NF4 in training, NVFP4 in serving).

### Readings, CHOSEN before the run

All rows by P1 (sign test p < 0.05 on the row's discordant items, Holm count beside).
Where a reference has three runs, each item is its mean over them, as `n2_items.py` does;
Q2 has one run.

1. **Did training help?** Q2 against base Qwen3.8: thinking on against the three low-effort
   runs (S1, r2, r3), thinking off against S1's one run. G6's rule under P1: it must win
   held_out and lose none of trap, trap_control, no_tool, task, rocky_task, alert and promql.
2. **Does it reach G6q?** Thinking on against G6q's three N1 runs, thinking off against its
   three L7 runs. It reaches G6q if it loses no row and the pooled sign test does not favour
   G6q at p < 0.05.
3. **Does it pass G6q?** Same comparison: at least one row won, none lost, and the pooled
   sign test favouring Q2 at p < 0.05. Only this makes Qwen3.8 + adapter the new candidate
   on quality alone; room (Qwen3.8 has 0.12x Lightning's KV) and speed (46.7 against ~220
   tok/s) are reported beside it, not folded in.

`probes/q2_compare.py` (`results/q2-compare.json`) computes these; before any Q2 eval ran it
fixed the one pooled test of readings 2 and 3 as: reaches if neither mode's pooled test nor
both modes' together favours G6q; passes only if both modes' together favours Q2.

### Q2 results, 2026-10-03

**The run:** 312 of 312 steps in 2.92 h on the laptop, no OOM and no guard abort; loss 1.33
on step 1 to 0.037 on step 312 (the last 10 steps' mean 0.149 at the end of epoch 1, 0.098 at
the end of epoch 2); the card peaked at 22.25 GiB (20.68 GiB allocated by torch) and 87 C,
its thermal target, below the guard's 90; the only throttle flag was `sw_power_cap`, the
175 W limit (`results/q2-train.json`). Adapter: `results/q2-train-adapter/` (gitignored), with
checkpoints at steps 52 to 312.

**Served on the desktop** at the pre-registered flags: 17.1 GiB of weights and LoRA, 1.15
GiB of KV, 23,130 tokens, 1.41 requests of 16,384; decode 38.4 tok/s at c=1 and 32.0 a
stream at c=16, against the base's 46.8 and 38.1 on the same card (r2). The seven sets took
767 s thinking on and 753 s thinking off; base Qwen3.8's low-effort pass took 1,551 s on
this card (r2).

**It no longer thinks.** With thinking on, every turn of v2 (263 of 263) opens with an empty
`<think>` block and goes straight to the tool call or answer: 86 tokens an item on v2
against the base's 535 (r2), 78 over all seven sets. G6q's data teaches that: its 891 thinking-on records carry no reasoning,
the label starts at `</think>`. G6q does the same (271 of 274 v2 turns empty, 85 tokens an
item, against base Lightning's 737), so the comparison with G6q is like for like.

| reading | thinking on | thinking off | verdict |
|---|---|---|---|
| 1. against base Qwen3.8 | P1 win 5, loss 0 (Holm 2 of 20); pooled 122/36 | P1 win 7, loss 0 (Holm 2 of 20); pooled 119/32 | **helped**: G6's rule passes in both |
| 2. against G6q (N1 x3 / L7 x3) | P1 win 1 (rocky_task 8/1, p 0.039), loss 0 (Holm 0 of 14); pooled 29/22, p 0.40 | P1 win 0, loss 0 (Holm 0 of 20); pooled 23/36, p 0.12 | **reaches G6q** |
| 3. passes G6q | | | **no**: both modes pooled 52/58, p 0.63 |

- Reading 1's wins are the flag rows (held_out, held_out2, rocky_held_out, seen_tool thinking
  on) and the traps (rocky_trap; with thinking off also trap, trap2 and trap3). No row is
  lost; task (4/5 thinking on) and the trap controls (0/4) lean to the base without reaching
  p < 0.05.
- Against base Lightning's N1 runs, thinking on: P1 win 4, loss 0, pooled 112/31.
- **Reasoning rows**, thinking on, in items (one run; three-run means for the rest):

  | | task | rocky_task | promql | alert | trap3 noticed |
  |---|---|---|---|---|---|
  | Q2 | 15 | 14 | 14 | 8 | 7 |
  | base Qwen3.8 low x3 | 17.67 | 13.33 | 13.67 | 9.00 | 7.33 |
  | G6q N1 x3 | 14.67 | 8.00 | 13.67 | 5.00 | 5.00 |
  | base Lightning N1 x3 | 13.00 | 12.67 | 12.00 | 6.00 | 3.67 |

  Q2 keeps base Qwen3.8's reasoning rows within a few items while gaining G6q's flag rows;
  G6q lost rocky_task in training (8.00 against Lightning's 12.67), Q2 did not.

**Verdict by the pre-registered rule: Qwen3.8 + adapter reaches G6q and does not pass it, so
it is not the new candidate on quality alone.** Beside it, not folded in: it holds 23,130 KV
tokens on a 3090 against Lightning's ~266k on one card, and decodes at 38 tok/s against
~225. G6q stays the candidate; Q2 is the stronger reasoner at the same flag accuracy.

## Memory tiers: RAM for the KV cache and for 70B weights, pre-registered 2026-10-03

David, 2026-10-03, after asking whether the nodes' system RAM can add to their VRAM: "lets
try 1. Add --kv-offloading-size to laptop Qwen3.8 serving and compare eval wall time ...
2. Check whether 4-bit weight streaming can train a 70B model on the laptop."

**Measured, both nodes** (`torch` copies of 1 GiB, best of 5, pinned host memory):

| | laptop RTX 5090 Laptop | desktop RTX 3090 |
|---|---|---|
| RAM | 61 GiB, 2 x 32 GB DDR5-6400 | 30 GiB, 4 x 8 GB DDR4-2133 |
| PCIe | Gen5 x16 | Gen3 x16 |
| RAM to GPU | 51.8 GB/s | 12.2 GB/s |
| GPU to RAM | 23.6 GB/s | 11.2 GB/s |
| copy inside VRAM | 717 GB/s | 806 GB/s |

### K1: a RAM tier for Qwen3.8's KV cache on the laptop

r3's command (`NODE=laptop`, util 0.90, fp8 KV, language model only, `PASSES=think-low`)
plus `--kv-offloading-size 24`: vLLM v0.29.0's native CPU offloading, 24 GiB of RAM. TAG
`q38-kvo`. It is a prefix tier, keyed by block hash (`OffloadingConnector`, which declares
hybrid-model support, `SupportsHMA`): a request whose prefix blocks were evicted from VRAM
loads them back from RAM instead of recomputing them. It does not let more requests run.

**Prediction: no gain.** Every Qwen3.8 serve so far logged a prefix-cache hit rate of 0.0%
(S1 589 lines, r2 170, r3 244), on the desktop's larger cache too, and none preempted a
request. vLLM sets Qwen3.8's attention block to 1,568 tokens so that a block's page matches
a DeltaNet state's ("Setting attention block size to 1568 tokens"); a hit needs a whole
such block and its saved state. A RAM tier holds what VRAM evicted; nothing hit before
eviction. **Reading:** the seven sets' summed `elapsed_s` against r3's 2,186 s (2,194 s by
the screen's clock) and the serve log's hit rate. Below 2,186 s by more than the
laptop-to-laptop spread we have no measure of yet would be a gain; the hit rate says why.

**Before the run, why the hit rate is 0.0%** (a smoke on the idle desktop, same flags plus
`--kv-offloading-size 8`, and r3's own records):

- The tier starts with Qwen3.8 (an 8.58 GB shared-memory buffer) and costs no VRAM: KV room
  stayed 1.63 GiB, 31,804 tokens.
- Prefix caching does work for Qwen3.8, a block at a time. The same 3,983-token prompt sent
  twice hit 3,136 tokens the second time, two whole 1,568-token blocks; the same system
  prompt with another question hit the same 3,136. All hits came from VRAM; the RAM tier
  stored the blocks (257 MB) and served none, since nothing had been evicted.
- The eval's prompts never fill one block. In r3, first turns run 437 to 937 tokens (median
  about 440) and later turns' medians 587 to 1,046; no prompt of the 478 items reaches 1,568.
  So no prefix is ever cached, in VRAM or RAM; the laptop's limit is the KV room for the
  requests running (3.6 at a time), which a RAM tier does not add.

**Amended before either arm ran: a control arm.** At launch, desktop apps held 268 MiB of the
laptop card against about 0.6 GiB during r3, and vLLM charges other clients' memory to KV, so
a run now gets more KV room than r3 had and could be faster for that alone. So K1 runs r3's
flags first, filed as repeat r4 (`TAG=q38-int4 REP=4`), then the same with the RAM tier
(`TAG=q38-kvo`), back to back; the reading is the RAM-tier arm against r4, with r3 beside.

**K1 result (MEASURED, 2026-10-03): no gain, as predicted.** Both arms 268 MiB of desktop
apps, KV 20,239 tokens, all 7 sets exit 0, no request preempted in either serve log:

| | r3 | r4 (control) | q38-kvo (RAM tier) |
|---|---|---|---|
| KV room, tokens | 19,275 | 20,239 | 20,239 |
| summed `elapsed_s` | 2,186 | 1,902 | 2,039 |
| tokens generated | 219,204 | 213,712 | 214,430 |
| generated per second | 100.3 | 112.4 | 105.2 |
| decode c=1 / c=16, tok/s | | 42.45 / 38.0 | 42.17 / 37.6 |

- The RAM tier served almost nothing: its hit rate ("External prefix cache hit rate") read
  0.0 to 2.3%, mostly under 1%; VRAM's read 0.0% in all 225 lines. Answers ran past 1,568
  tokens, so whole blocks were stored, but almost no later prompt shared them.
- At equal output (+0.3%) the RAM-tier arm ran 7% longer; the copies out to RAM are a
  plausible cost, but we have no laptop-to-laptop spread at equal KV to call 7% real.
- r4 ran 13% faster than r3 with 5% more KV room, the only difference in their flags being
  the desktop apps' memory. The laptop's eval time follows VRAM KV room (how many requests
  run at once), which a RAM tier does not add: closing apps did more than 24 GiB of RAM.
  Against the desktop's 1,558 s (31,804 tokens) the gap narrowed from 628 s to 344 s.

### W70: training a 70B with its 4-bit weights streamed from RAM

`probes/w70_stream.py` (its docstring has the design). The lab's earlier "70B CPU offload is
ruled out" came from accelerate keeping offloaded weights in bf16 (gpu-lab 3e41b1e); here
every layer is NF4 in pinned RAM and is copied to the card only while it computes, the next
layer's copy overlapping the current layer's work. Base: the GPTQ-INT4 Llama-3.1-70B the
pool already serves (no download), unpacked and re-quantized to NF4. LoRA and optimisation
as Q2; data G6q's records in Llama 3.1's template (1,248 render, 1,064,148 tokens, longest
1,800, none cut).

**Gates, in order:**
1. `SELFTEST=1` on the first 4 layers: the layer loop streamed equals it resident bitwise
   (loss and every LoRA gradient), and equals plain autograd to within 2% of the largest
   gradient; GPTQ's unpacked zeros are all 8 (it is symmetric).
2. The 80-layer run, `RESIDENT=0` (every layer streamed), `STEPS=6`: step 1 is the 8
   longest records, then 5 steps in G6q's epoch-0 order. A record's loss must be sane for
   an instruct model on chat text (below about 3 nats a token; a wrong unpack gives far more).
3. If 2 fits, `RESIDENT` raised to the most layers that fit, and the speed difference.

**Arithmetic before the run** (to be checked against the run, not trusted):
- an NF4 layer is 0.41 GiB (855.6M parameters at 4 bits plus an absmax byte per 64), so
  80 layers are 32.9 GiB of pinned RAM;
- each record copies every layer twice (forward, then backward), 65.7 GiB, about 1.4 s at
  51.8 GB/s, so about 11 s per 8-record step;
- the card holds about 9 GiB besides layers: lm_head 1.96, LoRA r16 (159.9M parameters,
  fp32 with gradients and 8-bit Adam) 1.5, the kept layer inputs at 1,800 tokens 2.2, one
  layer's working set about 1, two layers in flight 0.8, CUDA and desktop apps about 1.6;
  so about 25 to 30 layers could stay resident;
- compute about 2.5 times Q2's per token (70.6B against 27.8B parameters), so about 80 s a
  step and about 7 h for G6q's 2 epochs if the copies hide behind it.

**Reading:** it trains on this card if gates 1 and 2 pass with no OOM and no guard abort;
reported beside it are the peak memory, seconds a step, the copy volume, how much of each
streamed layer's time the copy adds, and the hours a full run would take.

**Gate 1 PASS, on the desktop 3090** while the laptop trained Q2 (its gpu-lab:training
build differs, its torch 2.13.0, transformers 5.16.1, peft 0.21.0 and bitsandbytes 0.50.2
do not; `results/w70-selftest-desktop.json`):

- With deterministic kernels (SDPA's math backend, `use_deterministic_algorithms`), the
  layer loop streamed equals it resident, and the loop equals plain autograd, bit for bit:
  loss 12.685362 (4 of 80 layers, so no real model) and every LoRA gradient.
- With the default kernels, as training runs, two resident runs differ by 0.0040 in the
  largest gradient (0.61) and resident against streamed by 0.0041: attention's backward is
  not deterministic, and streaming adds nothing to it.
- The first try's "bitwise" gate failed for that reason and was rewritten to measure the
  floor; it also showed `inject_adapter_in_model` leaves LoRA in bf16, so the probe now casts
  it to fp32 as Q2's `get_peft_model` does.

**The unpack is right** (`LOSSCHECK=4`, `results/w70-losscheck-desktop.json`): every one of
the 80 layers built in turn, 4 G6q records run through it, the layer dropped. Over all
tokens the records score 1.74 to 2.46 nats with the unpacked bf16 weights and 1.80 to 2.46
with NF4 (assistant tokens 1.50 to 3.75 and 1.55 to 3.86); a wrong unpack gives about 11.8
(ln 128,256 for a uniform guess) or more. GPTQ's zeros are all 8, as a symmetric checkpoint's
must be; layer 0's q_proj has a weight of 67.0, which these losses show is the checkpoint's.
NF4's error on that matrix is 3.2% of its norm.

**W70 result (MEASURED, 2026-10-03, laptop): a 70B QLoRA trains on the one 24 GB card.**
Gates 2 and 3 both ran 6 steps with no OOM and no guard abort (`results/w70-r0.*`,
`results/w70-r20.*`; max-power, 175 W cap, 84 to 87 C):

| | gate 2, `RESIDENT=0` | gate 3, `RESIDENT=20` |
|---|---|---|
| layers streamed from RAM | 80 (32.9 GiB pinned) | 60 (24.7 GiB pinned) |
| copied per step | 525.9 GiB | 394.5 GiB |
| GPU peak, nvml / torch allocated | 13.85 / 7.83 GiB | 22.07 / 16.04 GiB |
| process RAM peak, while loading | 52.1 GiB | 55.5 GiB |
| 6 steps, summed | 334.0 s | 334.3 s |
| full-run estimate (312 steps) | 4.62 h | 4.64 h |

- Losses per step 1.39 to 2.03 nats (gate 2's line: below about 3); the two runs agree to
  the third decimal, as gate 1's default-kernel noise floor allows.
- The copies hide behind compute entirely. A streamed layer takes 18.9 ms forward and 39.3
  ms backward against 18.6 and 39.0 resident (step 2, 583 tokens a record), and a 0.41 GiB
  copy is 8.5 ms at 51.8 GB/s. So keeping layers resident buys nothing: stream them all and
  keep the 8 GiB of VRAM.
- Against the arithmetic: layer size, pinned RAM and copy volume came out as computed; the
  card held 13.9 GiB besides the layers, not about 9 (torch reserves 12.2 GiB for 7.8
  allocated), so about 23 layers fit resident, not 25 to 30; steps ran 38 to 100 s, not
  about 80, and a full run is about 4.6 h, not 7.
- The tight resource is system RAM, not VRAM: loading peaks at 52 to 55 GiB of the
  laptop's 61 (the GPTQ shards are read while the NF4 copies are pinned). The desktop's 30
  GiB could not pin all 80 layers (32.9 GiB; arithmetic, not run).
- This is a fit-and-speed probe: no adapter was saved and nothing was evaluated. A full run
  needs an adapter-saving path and a Llama-3.1-70B eval plan before it means anything.

## L70: untrained Llama-3.1-70B on the seven sets, pre-registered 2026-10-03

David, 2026-10-03: "ok we can proceed with measuring untrained 70B. from there we can either
test serving the 70B on one card and then proceed to weakness targeting or proceed to weakness
targeting and serve the 70B but i do want to eventually try serving the 70B on the laptop".

W70 showed a 70B adapter can be trained on the laptop (about 4.6 h for G6q's data). L70 asks
whether one is worth building: how does the untrained 70B stand against G6q, the candidate,
and Qwen3.8-27B, the other base? If it is not clearly ahead somewhere that matters, weakness
targeting goes to G6q. David picks the branch after the result.

### Model and serving

- `hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4` @ 1b0ae7f9, the same revision in
  both nodes' caches, no download. This is the build the lab's September 70B evals used and
  W70 trained.
- Served across both cards by gpu-lab `bin/lab pool up`: vLLM 0.29.0 (`gpu-lab:vllm-ray`),
  pipeline parallel over the cable, the desktop 3090 and the laptop's 5090. Flags:
  `--enforce-eager` (the engine will not start without it: gpu-lab's pooled-70B notes),
  util 0.92 and a 512-token prefill batch (bin/lab's measured defaults), maxlen 8,192.
- **Why maxlen 8,192, not S1's 16,384:** at 4,096 the pool held 16,320 KV tokens
  (MEASURED, 2026-09), fewer than one 16,384-token sequence, so vLLM would refuse. In
  earlier runs of these sets the longest prompt was 2,508 tokens (G6q, rocky), so 8,192
  leaves every turn its full 512 tokens. The KV line the engine logs at this maxlen is
  recorded with the results.
- Laptop GPU apps closed before `pool up`: vLLM counts their memory against KV.

### Protocol

- **One pass: thinking off, 512 tokens a turn.** Llama 3.1 has no thinking mode. With
  `--thinking on`, `g7a_eval.py`'s `split_think` reads any reply without `</think>` as
  truncated mid-think, which would score every 70B answer as empty. `--thinking off` sends
  `enable_thinking: false`, which Llama's template ignores, and leaves the harness's own
  `split_think`.
- Otherwise S1's protocol: `probes/g7a_eval.py`, gated by `--selfcheck`, `--max-calls 3
  --temperature 0 --window 4000 --seed 20260923 --concurrency 16`, the promql set with
  `--promql-catalog` against the desktop's Prometheus. Labels `{set}-s1-l70-nothink`.
- Tool calls come back as Llama's JSON (`{"name": ..., "parameters": ...}`), which
  `research_eval.parse_tool_call` reads; it scored the September 70B runs. Tool output goes
  back through Llama's template as `ipython` turns.
- **Before the full run:** one warm-up request, discarded (an eager 70B's first request
  is slow); then a smoke run, `--limit 1` on v2, whose turns must show a parsed tool call
  and a final answer. If the smoke run fails, stop and fix the harness. Do not score.

### Comparators

1. **G6q, thinking off:** its three L7 runs, all seven sets. This is the like-for-like
   comparison: neither side thinks.
2. **G6q, thinking on:** its three N1 runs on N1's five sets (v2, rocky, promqlcat, alert,
   trap3). On v1 and general it has one run (G6's gate run).
3. **Qwen3.8-27B INT4, thinking on at low effort:** its three runs (S1, r2, r3), all seven
   sets.

Beside them, outside the rule: base Lightning (N1 x3 thinking on, five sets; one run
thinking off, seven sets) and the 70B's September v1 results.

Every row is judged by P1: a win or a loss only at two-sided sign p < 0.05 on the
discordant items, with each item's score the mean over that side's runs
(`s1_compare.rows_rep`). The Holm count is beside, over the 20 rows with per-item scores.
`general.correct_where_scorable` and `no_tool.correct_where_scorable` have none, so they
are reported as rates only. `probes/l70_compare.py` computes all of this and writes
`results/l70-compare.json`.

### Readings, CHOSEN before the run

The rows that matter are the reasoning rows, which training here has not been shown to
add: task, rocky_task, promql and alert, scored per item, plus general as a rate. G6q's
data reliably adds the flag and trap rows: Q2 and N4 both gained them. So a 70B loss to G6q
on those rows is expected, and an adapter would close it. Those rows are reported, but they
do not count against the 70B.

1. **Clearly ahead somewhere that matters, so a 70B adapter is worth building:** at least
   one of task, rocky_task, promql or alert is a P1 win against all three comparators, on
   the same row, and none of those four rows is a P1 loss against any of them.
2. **Otherwise: not clearly ahead.** Weakness targeting goes to G6q.

Reported beside the readings, not folded in:
- the pooled sign test per comparator;
- each comparator's full row table;
- the reasoning rows in items, beside N1's table;
- items that ended `truncated` (512 tokens hit) or `context_exhausted`;
- how often a tool was called on no_tool and general (September: 20 of 20 no-tool
  questions);
- wall time per set.

The weakness-targeting step that follows must pre-register "no row lost" on all seven sets
(memory: targeted data needs controls). That rule is not L70's.

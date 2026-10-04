# Training, testing and data: what to change next

Date: 2026-10-04. A synthesis of two reports written the same day:

- `docs/harness-audit.md` ("audit"): a read-only, CPU-only audit of our builders, trainer,
  eval and scorers. "Audit fix N" is its Top 10; "audit §N" is a section.
- `docs/training-practices-research.md` ("research"): a web survey of LoRA/QLoRA, data, hybrid
  thinking, tool use, preference/RL and eval practice. "Research #N" is its Top 10.

Label key: **[measured]** observed on our files or hardware; **[sourced]** from a cited
source in the research note; **[arithmetic]** computed here; **[inference]** judgement, not
tested. Nothing in this note was trained or evaluated; it plans work, it reports none.

## Bottom line

1. **The mechanics are sound.** Loss masking is correct on all 1,248 G6q records: 99,387
   trained tokens, all in assistant turns, none in tool output [measured, audit §4]. The chat
   template and tokenizer are byte-identical between the training checkpoint and the served
   one [measured, audit §5].
2. **The weak part is measurement.** The data that made G6q was written after reading
   per-item failures on the same items it was then judged on [measured, audit §8.5]. About
   10 adapters were ranked on those items [inference, audit §8.5]. On the rows that decide
   the choice, one adapter's run-to-run spread is as wide as the gate's band: G6q alert gave
   7, 3 and 5 of 9 on three identical runs [measured, `results/n1-summary.json`].
3. **So the order is: measure, then data, then recipe, then new methods.** More data
   cannot show up as progress until the eval can see it, and more of the same templates
   makes things worse (Tier 2.1). Real data volume comes from verified own-domain records
   plus licensed external sets used as controls and replay (see "Data volume").

## Tier 1: measurement integrity (before any new training)

### 1.1 A sealed test set

Audit fix 1 and §8.5; research #2 and §6.4.

- **Three disjoint pools.** The current 478 items become the **dev** set: keep using them
  for iteration, but stop calling them held out. A new **locked test** set is opened once
  per final candidate and never read item by item by whoever writes training data.
  Training data stays separate.
- **Who writes it** [inference]: a different generator, or a session with no access to
  `results/`. Same-author paraphrase leaks with zero string overlap [sourced, research
  §2.2, arXiv 2311.04850].
- **Size**: alert needs 30+ items to be able to show a regression at all (1.3).

### 1.2 Skeleton contamination check

Audit fix 2 and §1.3; research §2.2 and #6.

- **The exposure** [measured text, twin pairing is inference]: 6 of the 9 alert items and
  about half of the 18 promql items have a training twin that differs only in the metric
  name. Today's check (`g6q_build.py`) excludes metric names and Jaccard >= 0.5. It never
  looks at the shape of the query or rule.
- **The check**: mask metric names, durations and thresholds, then compare operator
  skeletons (`absent(...) for 5m`, `avg_over_time[10m] > N`). Add embedding
  nearest-neighbour search. Report "max similarity to train" for every eval item.
- **Make it mechanical**: the twin table in audit §1.3 is one reader's pairing (audit §10),
  so the first step is a script, not more reading.

### 1.3 Repeats, aggregation and the decision rule

Audit fix 3, §6 and §8.4; research #1, §6.2 and §6.3.

**Rule.**
- Run at least 3 repeats per arm. Average each item over its repeats **before** the sign
  test [sourced, research §6.2].
- Apply Holm across a family of rows defined before looking.
- Pre-register one pooled primary metric and report it with a CI [sourced, research #1:
  Miller arXiv 2411.00640, Bowyer arXiv 2503.01747].
- Temperature 0 is not deterministic [sourced, research §6.2]: G6q gave 7, 3 and 5 on
  alert under identical settings [measured].

**Gaps in today's tools** [code, audit §8.4]:
- `pair_items.py` marks raw p < 0.05 and computes **no Holm**. With 20-25 rows, one raw "<"
  is expected under the null.
- v1 (held_out, the gate's must-win row) and general have a **single** thinking-on run per
  adapter [measured, audit §6].

**What each set can show** [arithmetic]: exact two-sided sign test. Holm's strictest step
over 7 sets is 0.05 / 7 = 0.0071.

| set (items) | smallest passing split | why it matters |
|---|---|---|
| any | 9 discordant items, 9-0 (p 0.0039) | 8-0 gives p 0.0078: below 9 discordant items, nothing passes |
| alert (9) | only 9-0 | 8-1 gives p 0.039; a real 2-item regression is invisible |
| trap3 (12) | 11-1 (p 0.0063) | 10-2 gives p 0.039 |
| promqlcat (18) | 16-2 (p 0.0013), all 18 discordant | 15-3 gives p 0.0075 |
| alert enlarged to 30 | 24-6 (p 0.0014) | the reason for 1.1's 30+ |
| 20 discordant pairs | 17-3 (p 0.0026) | 16-4 gives 0.012 and 15-5 gives 0.041: these pass raw, fail Holm |

**Pseudo-replication** [inference, audit §8.4]: items come in clusters of 6 per tool with
shared templates, so they are not independent. The pooled primary metric with a CI over
clusters is the safer headline.

### 1.4 Item manifest and cwd assert

Audit fix 7, §8.3.

- **The cwd bug** [measured]: `research_eval.py` builds items from the host's `--help` in
  the caller's cwd. Outside a git repo, v1 is 166 items, not 158. Of the 154 ids both lists
  share, 39 name a different flag and question. `pair_items.py` joins by id, so it would
  silently pair different items.
- **Existing comparisons are safe** [verified, audit §8.3]: all 418 result files are
  158/134/102/18/45/9/12.
- **Fix:**
  - Write each set's items once to JSON, and have every run load the file by hash.
  - Assert the cwd is inside a repo (`l70_eval.sh` already `cd`s; `g6_eval.sh` and
    `noise_eval.sh` do not).
  - Record the rocky image and promtool image digests in every result file.

### 1.5 Scorer hardening

Audit fix 6, §8.2. Each case below is measured with synthetic answers through the real
`score()`.

**Wrong answers that pass:**
- **promql:** "It might be 20, 40, 52 or 55 degrees" with no tool call scores correct.
- **promql nodata:** "the queue is empty" scores correct because the word "empty"
  matches.
- **trap:** "No problem! Use `cp --parallel-copy`" scores as a denial.
- **flag:** "Do not use -r" scores a hit for `-r`.
- **general:** "144 divided by 12 is 13" passes `\b12\b`.
- **alert:** for a 5m rule asked `> 85`, `for: 8m` passes, and so do `> 71` and `> 89`.

**Right answers that fail:**
- "Six hundred thirteen."
- "7/8 is 87.5%."
- "Yes, cp does not have a --parallel-copy option."

**Fix**: turn the `scorer_probe.py` cases into a scorer test file, with both directions as
regression cases. Then tighten:
- one number per promql answer;
- nodata needs an explicit no-data statement;
- negation handling for denials and flag hits;
- exact duration and threshold match in alert rules.

Re-score the existing result files wherever they store what the scorer needs, and report
any rank that changes. promql truth is read live after the answer, so those rows may not be
re-scorable [inference].

### 1.6 Keep the filter and the scorer apart; pin lab state

Audit fix 10, §8.5.

- **Shared code** [code]: training records are filtered with the eval's own functions
  (`grounded_token`, `GLOBAL_DENIAL`, `score_alert`). The data is then guaranteed to pass
  the scorer, scorer bugs included. Give the builders their own validators (promtool, a
  fixture Prometheus, `--help` text), not the scorer.
- **Live-lab dependence** [code]: the promql no-data items are right only while the vLLM
  targets are down, and the training records teach that same lab state. Move promql truth
  to a fixture Prometheus with known series, or check and record target state per run.

### 1.7 Report reasoning, not just answers

Audit §3; research §3.

- **0 of 2,424** G6q assistant turns hold reasoning [measured]. G6q's "thinking on" is
  thinking off with a bigger budget.
- **Add to every result**: the valid-reasoning rate (a non-empty, closed think block) per
  mode [sourced, arXiv 2605.21127].
- **Add a reasoning-required held-out slice.** The current `general` set ("What is 3
  cubed?") cannot show a reasoning regression [inference, audit §3].

### 1.8 Correct the record (done in this commit)

`lightning-training.md` quoted the gate run (r1) for G6q and G6u. Both rows now carry the
N1 three-run means [measured, `results/n1-summary.json`]:

| | row | gate run | N1 mean of 3 |
|---|---|---|---|
| G6q | promql | 15/18 (0.833) | 13.7/18 |
| G6q | alert | 7/9 | 5.0/9 |
| G6u | task | 19/20 (0.95) | 17.7/20 |
| G6u | trap3 noticed | 9/12 (0.75) | 8.0/12 |

The L7/L8 choice (G6q by default, G6u as backup) rests on item-level comparisons with three
repeats a side, so this correction does not change it. Both adapters were still selected on
the dev items, so the locked test (1.1) is what confirms either one [inference].

## Tier 2: data hygiene

### 2.1 Dedupe and template caps

Audit fix 5, §2; research #6, §2.1 and §2.2.

**What the data looks like** [measured]:
- 1,248 records, 157 of them exact duplicates (1,091 unique).
- Six prompts appear 10 times and four appear 12 times. web_research has 6 unique prompts
  in 60 records; conversational_replay has 4 in 48.
- 544 records (43.6%) answer "Based on `x --help`".

**The arithmetic**: 12 copies x 2 epochs = 24 passes over one answer. Training loss reached
0.0016 [measured, audit §7]. That is memorised templates, not signal [inference; research
§1.4].

**Pipeline**, in order:
1. exact dedup;
2. MinHash near-dup;
3. embedding dedup;
4. per-template caps (at most 2-3 records per prompt template, and a cap on any one
   answer opener) [sourced recipe, research §2.2; numbers are inference].

### 2.2 Controls and a 7-set regression gate

Audit fix 9, §2; research #4, §2.4.

- **Our own lesson** [measured, 8B v2]: targeted data without controls took alert rules
  from 78% to 11%.
- **G6q's gaps** [measured, audit §2]:
  - The promql tool appears in only 58 of 1,248 records. No record offers promql and
    rightly answers with bash or no tool.
  - There are no system messages and no second user turns.
  - Thinking-off no-tool answers appear only in 11 alert records.
- **Rule**:
  - Every weakness-targeted slice gets at least as many control records from the other
    task types.
  - Every tool appears in records where it is offered and correctly **not** used.
  - Add system-message and multi-turn records.
  - The gate is all 7 sets, not the targeted row.

### 2.3 Verified targets

Research #6, §2.3; audit §9.

- **Every target must pass a checker that is not the scorer:**
  - promtool parse plus a result match against a fixture Prometheus;
  - flags present in the captured `--help`;
  - trap answers checked by rule.
- **Rejection-sample the model's own outputs against those checkers.** This is the
  strongest data lever available [sourced technique, research §2.3; inference for fit].
- **Fix web_search** [measured, audit §9]: it succeeds in 60 G6q records, but in the eval
  and the lab it is unavailable. The adapter is taught that search returns an answer.
  Either make those records fail like the lab does, or drop them.

### 2.4 Packing, masking and template tests

Research #8, §2.5; audit §4 and §5.

- Masking is already verified [measured]. Keep `maskcheck.py` as a test, run on every build.
- **Template parity.** G6u matched vLLM's prompt token counts on 502 of 502 turns; G6q
  has no such check [measured, audit §5]. Make the vLLM `/tokenize` diff on 20 items a
  build step.
- **Packing.** The trainer computes a per-record loss over 8-record steps
  [code, audit §4], so it does not appear to pack [inference]. If packing is ever added, the
  Mamba and DeltaNet layers need their recurrent state reset at each boundary
  [sourced, research §2.5]. Guard it with a "packed loss = unpacked loss" unit test before
  first use.

## Tier 3: training recipe

### 3.1 Validation and early stopping

Audit fix 8, §7.

- **Today** [measured]: no validation split, no held-out loss, no early stopping. The
  step-156 checkpoint is written and never evaluated.
- **Do**:
  - Hold out 5-10% of records, **grouped** by spec and tool so that a held-out record has
    no twin in training.
  - Log validation loss every 52 steps (`q2_train.py` already checkpoints at that cadence).
  - Evaluate the epoch-1 checkpoint on dev.
- **Also** [code, audit §9]: `g6_train.py` exits 0 after a guard abort. Make it exit
  non-zero, as `q2_train.py` does.

### 3.2 Learning-rate sweep, then seeds

Research #5, §1.1 and §1.4; audit §6.

- **LR is the dominant knob** [sourced, research §1.1, arXiv 2601.22708]. Sweep
  {5e-5, 1e-4, 2e-4}, one seed each, then 3 seeds of the winner.
- **Keep the rest fixed** [sourced, research §1.1]: r=16, alpha=32, all linear layers,
  effective batch of at most 32, at most 3 epochs.
- **Run on Lightning first** (3B active): it is cheaper per run [inference].
- **Seed variance is unmeasured** [measured: every adapter is one seed-0 run, audit §6].
  Until it is measured, a gap between two adapters is not attributable to their data.
- **Check first** [unknown, research §1.3]: does our Lightning path put LoRA on the expert
  weights, or only on attention and shared layers?

### 3.3 Reasoning mix

Audit fix 4, §3; research #3, §3.

- **G6q trains 0% reasoning examples** [measured]. Unsloth's guidance for hybrids is at
  least 75% reasoning examples to keep reasoning [sourced; a vendor heuristic, not a
  measured optimum]. Answer-only SFT suppresses valid traces while answers still look fine
  [sourced, arXiv 2605.21127].
- **G6t/G6u show that traces train** [measured]: G6u reasons on 555 of 994 thinking-on
  turns.
- **First decide whether reasoning is needed at all** (needs David, below). If it is,
  either follow the vendor's 75%, or pre-register a 25/50/75 sweep gated on the
  valid-reasoning rate (1.7). Use traces the model produced itself and that pass the 2.3
  checkers.
- **If it is not needed**: serve thinking off and stop reporting "thinking on" as the
  primary mode.

## Tier 4: later, once Tiers 1-3 are in place

### 4.1 When2Call and DPO/KTO on our own confabulations

Research #7, §4.2 and §5.1.

- When2Call: CC-BY-4.0; 15,000 SFT, 9,000 preference pairs, and a 3,652-item MCQ test
  [sourced, card read].
- **Why it fits**: its authors found preference optimisation beat SFT on call / ask /
  decline choices, which is our trap category [sourced, arXiv 2504.18851].
- **Pairs**: chosen = verified behaviour; rejected = what our adapters actually produced in
  eval [inference]. Re-render into each model's own template.
- Keep When2Call's test split out of training, as an external eval.
- **Needs David's go to download.**

### 4.2 NF4 adapter vs a better base; batch invariance

Research #9, §1.3 and §6.2.

- **QLoRA on the Qwen3.5 family**: Unsloth says it is "not recommended", and lists
  56 GB for 27B bf16 LoRA [sourced]. That does not fit 24 GB [arithmetic].
- **Cheap check**: serve the NF4-trained adapter over the NF4 base and over the FP8 base,
  and compare on dev.
- **`VLLM_BATCH_INVARIANT=1`** needs compute capability 8.0 or higher [sourced]. Both cards
  qualify: sm_86 is 8.6, sm_120 is 12.0 [arithmetic]. Whether it holds with hybrids plus
  LoRA on 0.29.0 is unknown.

### 4.3 GRPO pilot

Research #10, §5.2.

- **Only after Tiers 1-3.** Use TRL with `dr_grpo` [sourced].
- **Rewards**: the 2.3 checkers (promtool plus a fixture result, a flag in `--help`, trap
  class by rule).
- **What it can and cannot do**: RLVR sharpens what the base already samples; it does not
  add facts [sourced, Yue et al. 2025].
- **Where to run it**: Qwen3.8-27B with colocated generation does not fit 24 GB
  [inference]. Start on a smaller model.

## Data volume: where more data comes from

"As much data as possible" holds only for data that is verified, deduplicated and balanced
[inference; research §2.1].

**Own-domain, no download:**
- rejection-sampled, checker-verified records (2.3);
- controls (2.2);
- traces the model produced itself (3.3).

**External, each needing David's go:**

| dataset | license (2026-10-04, verify the card) | size | use |
|---|---|---|---|
| nvidia/When2Call | CC-BY-4.0 | 15K SFT, 9K pref | call/ask/decline; DPO (4.1) |
| nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1 | CC-BY-4.0 | 31,111 | terminal tasks; prompts average ~39,900 characters, so as task ideas or a short subset |
| nvidia/Nemotron-Agentic-Tool-Use-v1 | CC-BY-4.0 per a mirror only | 335,122 | multi-turn replay sample; verify the license on HF |
| nvidia/Nemotron-Post-Training-Dataset-v2 | mostly CC-BY-4.0; gated; Qwen/DeepSeek terms on redistribution | 1M+ | reasoning on/off replay (3.3) |
| Team-ACE/ToolACE | Apache-2.0 per summary | ~10K | function-calling controls |
| NousResearch/hermes-function-calling-v1 | Apache-2.0 per summary | ~12K | controls and error turns; re-render |
| Salesforce/xlam-function-calling-60k | CC-BY-NC-4.0, gated | 60K | private research only; not before Tier 2 |
| BFCL | Apache-2.0 | n/a | **eval only**, never train |

No openly licensed PromQL or alert-rule dataset was found [unknown; not exhaustively
searched]. Prometheus docs and our own verified pairs remain the source.

## Not recommended

Research §"Not recommended":
- NEFTune: no evidence it helps exact-format output.
- DoRA, PiSSA, OLoRA: small or inconsistent gains; PiSSA rewrites the base weights.
- Rank above 16; attention-only LoRA.
- xLAM before the Tier 2 fixes.
- An LLM judge on rows a program can check.

## Judge pass on the two reports

| claim | verdict |
|---|---|
| G6q/G6u gains quoted from the gate run | CORRECTED: N1 means (1.8) |
| eval size 490 | CORRECTED: 478 in 7 sets |
| research §6.3 set sizes | CORRECTED |
| Unsloth's >=75% reasoning vs G6q's 0% | TENSION: follow the vendor or pre-register a sweep (3.3) |
| Unsloth: QLoRA not recommended for Qwen3.5 | CONFIRMED |
| ~56 GB for 27B bf16 LoRA | CONFIRMED |
| Unsloth disables router fine-tuning by default | CONFIRMED |
| When2Call sizes and license | CONFIRMED |
| the audit's measured counts | CONFIRMED |
| sign-test power table (1.3) | CONFIRMED by exact computation |

## Open unknowns

- Does our Lightning path train the expert layers (3.2)?
- NF4-trained vs bf16-trained Qwen3.8 adapter quality (4.2).
- vLLM 0.29.0 batch invariance with hybrids plus LoRA (4.2).
- GRPO on a hybrid MoE in 24 GB (4.3).
- An open PromQL dataset.
- NEFTune replications.

## Needs David

- **Download go**, per dataset: When2Call first (4.1); any other row in the volume table.
- **Is reasoning needed?** If yes, train the mix (3.3). If no, serve thinking off and drop
  thinking-on as the primary mode.
- **Who writes the locked test set** (1.1), and an agreement that its items are never read
  per item during data work.
- **Merges**: this branch, `worktree-s1-screen` and `lab-alerts` are proposals; none is
  merged.

## Suggested first week [inference]

1. Tier 1, CPU only:
   - the manifest and cwd assert (1.4);
   - scorer tests and fixes, then re-scoring the existing results (1.5);
   - the skeleton contamination script (1.2);
   - Holm in `pair_items.py` (1.3).
2. Tier 2, CPU only: dedupe, caps, controls and the web_search fix on a copy of the G6q
   data (2.1-2.3).
3. The first GPU work: a 3-point LR sweep on Lightning with a grouped validation split
   (3.1, 3.2), judged on dev with 3 repeats.
4. Write the locked test set in parallel. Use it only when a candidate is final.

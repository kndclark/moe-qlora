# Training, testing and data: what to change next

Date: 2026-10-04. A synthesis of two reports written the same day:

- `docs/harness-audit.md` ("audit"): a read-only, CPU-only audit of our builders, trainer,
  eval and scorers. "Audit fix N" is its Top 10; "audit §N" is a section.
- `docs/training-practices-research.md` ("research"): a web survey of LoRA/QLoRA, data, hybrid
  thinking, tool use, preference/RL and eval practice. "Research #N" is its Top 10.

Label key: **[measured]** observed on our files or hardware; **[sourced]** from a cited
source in the research note; **[arithmetic]** computed here; **[inference]** judgement, not
tested. Nothing was trained for this note. The one evaluation is the decision rule's dry
run (1.3), on existing result files plus the repeats it needed.

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
3. **G6q is not better than the base everywhere.** On rocky_task it scores 8.0 of 20
   against the base's 12.7, outside both arms' spread; on alert it is 5.0 against 6.0,
   inside the noise [measured, N1 means, `results/n1-summary.json`]. Every comparison
   therefore carries the base as an anchor (1.3), and the locked test can return "serve the
   base" (see "Exit criteria").
4. **So the order is: measure, then data, then recipe, then new methods.** More data
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
- **Where items come from** [inference]: prefer items with an outside ground truth:
  `--help` text of tools absent from training, examples from the Prometheus docs, and real
  lab questions (David's own, or LiteLLM request logs if they are kept). A model-written
  item is the last resort. Every item passes the 1.2 check against train **and** dev
  before sealing.
- **Where it lives** [inference]: both repos are public, so the test set never enters
  either one. Keep it on the desktop outside any checkout, commit only its SHA-256
  manifest, and append one line per opening (date, candidate, result file) to a ledger.
  That makes "opened once per final candidate" checkable.
- **Retire it after a few openings** [inference]: each opening's decision feeds back into
  the next round's choices, so the test slowly turns into dev. Write a fresh one after 3
  openings, and say in each result how many openings came before it.
- **Size** [arithmetic and inference]: 30+ items per task kind, about 180-200 in all.
  Alert at its dev size of 9 cannot show a regression under a sign test (1.3); at 30 the
  row margin is 3 items. Size against the margin each row must detect, not against what is
  easy to write.
- **promql and alert wait for the fixture** [inference]: their truth is read from the live
  lab today (1.6), so a sealed item could change its answer after sealing. Seal those two
  kinds once 1.6's fixture Prometheus exists.
- **The sealing check reports counts only**: the 1.2 check on test items prints how many
  items exceed each similarity threshold, never the items, so running it leaks nothing to
  whoever writes training data.

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

**Rule** (implemented in `probes/decision_rule.py`).
- Run at least 3 repeats per arm. Average each item over its repeats **before** any
  comparison [sourced, research §6.2].
- **Anchors**: every comparison includes the base and the adapter currently served, on the
  same server config. "Better than G6q" is not "better than the base" (bottom line 3).
  `decision_rule.py --served ARM` prints the served arm's CI against the pick whenever
  they differ.
- **The row is the unit.** A row is one split under one headline metric, as
  `noise_summary.py` defines them (e.g. rocky / rocky_task / hit_and_grounded). Metrics
  where lower is better (a control's denial rate, over_trigger, fabricated) enter as
  1 - rate, so higher is always better. An item a metric cannot score is left out of that
  row. no_tool's correctness is a summary rate (`correct_where_scorable`); per item it is
  the `correct` field, which a per-item reader must map by name or it silently drops the
  row [measured: 0 of 37,355 items in the 514 files `results/**/research-eval-*.json`
  store it, 2026-10-04].
- **One pre-registered primary metric**: each task kind (`KIND`: flag, trap, no_tool,
  live for promql, alert, trap3) scores the mean of its rows. The primary metric is the mean of the
  six kinds. Its 95% CI comes from a paired bootstrap over tool clusters [sourced method,
  research #1: Miller arXiv 2411.00640, Bowyer arXiv 2503.01747; the weighting is
  inference]. Pooling raw items would hand every decision to the flag-lookup sets: v1, v2
  and rocky are 394 of 478 items (82%), alert under 2% [arithmetic].
- **Wins and regressions use different rules** [inference]:
  - **win**: an arm beats the base when its Holm-adjusted bootstrap p < 0.05 across the
    arms compared with the base, with a positive difference. The printed CIs are
    unadjusted. A claim that one row improved needs
    Holm over a row family fixed before looking.
  - **regression**: no Holm. Holm guards against false wins; applied to regressions at
    these sizes it passes almost anything (alert must go 9-0 to fail). Flag a **row**
    whose item-averaged drop against the base exceeds both max(2 items, 10% of the row)
    and the larger of the two arms' range across repeats. Any flag blocks promotion until
    it is explained. Flags are raised against the base only: the base is the floor, and an
    adapter's gains over it are weighed by the primary metric, not protected row by row.
  - **Why per row, not per set**: G6q's rocky_task loss (4.67 items of 20) is outweighed
    inside its own set by gains on rocky's other rows. The set nets a gain of 15.7
    item-rows, so a set-level margin flags nothing [measured, dry run below].
  - **Why the range condition**: run-to-run spread alone fires the margin. Set each
    arm against itself, one run against another, and the margin fires on 12 of 396
    ordered row pairs (3.0%; base 4, G6q 1, G6u 7), about 0.67 false flags per 22-row
    comparison of single runs [measured, 7 sets]. Means of 3 repeats shrink that spread,
    and the range condition asks the drop to exceed what the arms' own repeats show. In
    the dry run below the margin alone and the full rule raise the same one flag in 44
    row comparisons [measured]: a guard that did not bind here. Its price: a row as
    unstable as G6q's alert (7, 3 and 5 of 9) hides a 4-item drop, one reason alert is
    resized to 30+ (1.1).
- Temperature 0 is not deterministic [sourced, research §6.2]: G6q gave 7, 3 and 5 on
  alert under identical settings [measured]. The spread comes from batch composition, which
  serving has too [inference], so the repeats measure a real property. Keep the eval's
  concurrency fixed across arms and record it.

**Dry run on dev** [measured: `decision_rule.py --served g6q g6q g6u`, thinking on at
4,096 tokens, 3 repeats of all 7 sets per arm; 545 item-rows, 22 rows, 149 tool clusters]:

| arm | primary | flag-lookup | trap | no_tool | live | alert | trap3 | completion tokens / item | eval minutes |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.695 | 0.697 | 0.588 | 0.899 | 0.667 | 0.667 | 0.653 | 625 | 88.4 |
| G6q | 0.811 | 0.880 | 0.964 | 0.996 | 0.759 | 0.556 | 0.708 | 77 | 16.7 |
| G6u | 0.843 | 0.900 | 0.807 | 0.974 | 0.852 | 0.778 | 0.750 | 381 | 77.7 |

- Against the base: G6q +0.116 [+0.043, +0.187], Holm 0.005; G6u +0.149 [+0.106,
  +0.196], Holm 0.001. Both wins hold.
- Row flags: G6q one, rocky_task (20 items): base 13, 13, 12; G6q 9, 7, 8; drop 4.67,
  range 2. G6u none.
- **The rule serves G6u.** G6q is blocked by its flag, not by its score: G6u - G6q is
  +0.033 [-0.022, +0.094], p 0.24, not distinguishable.
- **Cost** [measured, same runs]: G6q writes 5.0x fewer completion tokens per item than
  G6u and 8.1x fewer than the base, never hit `max_tokens` (base 12 turns, G6u 7) and
  finished the 21 runs in 21% of G6u's time. Summed over all sets, tokens per item vary
  between repeats by up to 5.9% of the mean (base 608-645, G6q 76-77, G6u 376-386); one
  set alone varies by up to 52% (G6u general, 51-84). Item and cluster mix move the ratio too, so
  the rule tests cost at the bootstrap bound, not the point ratio: G6q's tokens are 0.20x
  G6u's, 0.22x at the 95% bound [measured]. Tokens are measured on the eval's task mix,
  not on the lab's traffic; the ratio may differ there [inference]. Eval minutes include
  one base v1 run of 207 s against 507-535 s for the other two, unexplained; tokens are
  unaffected.
- **A cost win needs non-inferiority** (step 2 of the decision tree). With G6q's flag
  cleared by hand on a scratch copy [measured], both arms qualify, but G6q - G6u has a
  lower bound of -0.094, beyond the default margin of -0.02, so the rule still serves
  G6u. At a margin of 0.10 it serves G6q. On today's dev numbers G6q needs both its flag
  explained and a margin above 0.094: the margin is David's call (Needs David).
- Without the alert set (9 items) G6q - G6u is [-0.051, +0.059] [measured,
  `--allow-partial` with alert hidden]: the alert row carries G6u's lead (+0.222 on one
  kind of six, +0.037 against a +0.033 total; G6q leads trap by 0.157). A 9-item row
  deciding the served model is the case for 1.1's 30+.

**Dry run, thinking off** [measured: `MODE=nothink decision_rule.py --served g6q g6q g6u`,
3 repeats of all 7 sets per arm; the same 545 item-rows and 149 clusters]:

| arm | primary | flag-lookup | trap | no_tool | live | alert | trap3 | completion tokens / item | eval minutes |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.570 | 0.468 | 0.563 | 0.994 | 0.241 | 0.556 | 0.597 | 294 | 34.5 |
| G6q | 0.852 | 0.887 | 0.971 | 1.000 | 0.852 | 0.667 | 0.736 | 76 | 16.5 |
| G6u | 0.874 | 0.871 | 0.977 | 0.990 | 0.741 | 0.667 | 1.000 | 79 | 17.5 |

- Both beat the base (G6q +0.282, G6u +0.304, Holm 0.001) with no row flag. **The rule
  serves G6u**: G6q - G6u is [-0.086, +0.035], not shown non-inferior, and G6q writes
  0.96x G6u's tokens (0.99x at the bound), so there is no cost case either.
- G6u is one item from a flag on v2 task (20 items): base 16, 13, 16; G6u 14, 12, 13; a
  drop of exactly 2.0 against a threshold of more than 2. On the three two-repeat subsets
  the rule serves G6u, G6q, G6u: without r2, base's dip leaves, the drop is 2.5 and the
  range 1, so G6u is flagged [measured, `REPS`]. That is the case for three repeats.
- **Off against on, per arm** [measured: `probes/mode_compare.py ARM`, this rule with the
  arm's thinking-on run as the anchor]: base -0.125 [-0.181, -0.065], 6 row flags; G6q
  +0.042 [-0.021, +0.105], no flag, 76 against 77 tokens; G6u +0.031 [-0.035, +0.099],
  one flag (v2 task 19, 17, 17 to 14, 12, 13), 79 against 381 tokens. On dev, thinking
  lifts the base and does not measurably lift either adapter, and it costs G6u 4.8x the
  tokens. **G6q's 5.0x cost lead over G6u exists only with thinking on.** These sets
  have no reasoning-required slice (1.7), so this shows the eval does not need thinking,
  not that the lab's traffic does [inference].

**Gaps in today's tools** [code, audit §8.4]:
- `pair_items.py` marks raw p < 0.05 and computes **no Holm**. With 22 rows on the 7 sets
  [measured], about one raw "<" is expected under the null [arithmetic].
- v1 and general had a single thinking-on run per adapter [audit §6]; base, G6q and G6u
  now have three [measured, fixed in this revision]. G6t still has one.

**What a per-row or per-set win can show** [arithmetic]: exact two-sided sign test. Holm's
strictest step over a family of 7 is 0.05 / 7 = 0.0071.

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

**Clusters on the locked test** [measured, inference]: a row with few clusters drops out of
some resamples, and its kind is then scored from its other rows. On dev,
v1/seen_tool/hit_and_grounded has 4 clusters and is absent from 34 of 2,000 resamples
(1.7%); no other row drops out and no kind is ever lost. The script falls back to the item
id when an item names no tool, which counts each such item as its own cluster. So the
locked test's manifest carries a cluster field for every item, and every row has at least
10 clusters.

### 1.4 Item manifest and cwd assert

Audit fix 7, §8.3.

- **The cwd bug** [measured]: `research_eval.py` builds items from the host's `--help` in
  the caller's cwd. Outside a git repo, v1 is 166 items, not 158. Of the 154 ids both lists
  share, 39 differ in flag or question (23 in the flag). `pair_items.py` joins by id, so it
  would silently pair different items.
- **Existing comparisons are safe** [verified, audit §8.3]: all 418 result files are
  158/134/102/18/45/9/12.
- **Fix:**
  - Write each set's items once to JSON, and have every run load the file by hash.
  - Assert the cwd is inside a repo (`l70_eval.sh` already `cd`s; `g6_eval.sh` and
    `noise_eval.sh` do not).
  - Record the rocky image and promtool image digests in every result file.
  - Record the training data's SHA-256, the git commit, the seed and the LoRA target regex
    in every adapter directory, and copy them into the result of any eval that serves the
    adapter. `g6q-train.json` names the dataset path but holds no content hash [measured].

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

**Then measure the scorer, not only test it** [inference]: a test file holds only the cases
someone thought of. Hand-label a stratified random sample (about 20 rows per set, both
arms) and report the scorer's agreement per set beside every result; redo it after each
scorer change. A set whose agreement is low cannot carry a decision.

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
- **Add cost per answer**: output tokens, wall seconds, the rate of hitting `max_tokens`
  and the rate of malformed tool calls, per mode. Whether to think (3.3, the router study)
  is a cost question as much as a score question [inference].
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

The L7/L8 choice (G6q by default, G6u as backup) rested on item-level comparisons with
three repeats a side, so this correction alone does not change it. 1.3's rule, run on the
same dev sets, does: it serves G6u, because G6q's rocky_task drop is a row flag (1.3 dry
run). G6q's case is cost: 77 completion tokens per item against G6u's 381, at a primary
metric the dry run cannot separate. Which to serve until the locked test is David's call;
both were selected on dev, so the locked test (1.1) confirms either one [inference].

## Tier 2: data hygiene

### 2.1 Dedupe and template caps

Audit fix 5, §2; research #6, §2.1 and §2.2.

**What the data looks like** [measured]:
- 1,248 records, 157 of them exact duplicates (1,091 unique).
- Six prompts appear 10 times and four appear 12 times. web_research has 6 unique prompts
  in 60 records; conversational_replay has 4 in 48.
- 544 records (43.6%) say "Based on `<source>`" in an answer; 508 open with it, and 329
  cite `--help` literally [measured].

**The arithmetic**: 12 copies x 2 epochs = 24 passes over one answer. Epoch-2 step loss
averaged 0.17, with single steps as low as 0.0016 [measured, `results/g6q-train.json`].
Low, spiky loss on repeated templates is consistent with memorisation; it does not prove
it [inference; research §1.4].

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
  - **A measured regression to repair first**: rocky_task, 8.0 against the base's 12.7 of
    20 [measured, N1].
- **Rule**:
  - Every weakness-targeted slice gets at least as many control records from the other
    task types.
  - Every tool appears in records where it is offered and correctly **not** used.
  - Add system-message and multi-turn records.
  - The gate is every row of all 7 sets, not the targeted row, under 1.3's regression
    rule.
  - **Teach behaviour, not facts.** Records teach when to call a tool, the answer's form
    and when to deny. Facts the model cannot look up at inference (metric names, flag
    spellings) stay in retrieval and tools [sourced, research: Gekhman et al. EMNLP 2024;
    matches our Phase 2b confabulation result].
  - Judge the Tier 2 changes as **one bundle** against G6q at the same two seeds (0 and
    1), otherwise identical (3.2). Ablate only if a row is flagged [inference: each
    ablation costs a training run plus its eval].

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

- **Seeds first.** Seed variance is unmeasured [measured: every adapter is one seed-0 run,
  audit §6]; until it is, a gap between two adapters is not attributable to their data.
  Retrain G6q's exact recipe at seeds 1 and 2. G6q was the best of about 10 adapters on
  the dev items, so its dev score is biased upward [inference]; the reseeds show both the
  seed spread and how much of G6q's margin survives.
- **The reseeds need one code change first** [code]: `g6_train.py` hard-codes
  `EPOCHS, LR, PER_STEP, RANK, SEED = 2, 1e-4, 8, 16, 0` (line 96), and `probes/gpurun.sh`
  passes an explicit `-e` list that has no SEED or LR. Read SEED and LR from the
  environment with today's values as defaults, add `-e SEED -e LR` to `gpurun.sh`, and
  write the dataset's SHA-256 into the result (1.4). Then, from the repo root:

      SEED=1 DATASET=/out/research_dataset_g6q.json RENDER=think GUARD=hw MAX_LEN=2048 \
        probes/gpurun.sh g6q-seed1-train /probes/attn_bf16.py g6_train.py g6q-seed1-train

  This is the command `lightning-training.md` gives for G6q, plus SEED and a new label
  [measured: `g6q-train.json` records seed 0, max_len 2048, guard hw, render think].
- **Cost per run** [measured; arithmetic]: ~50 min of training (2,970 s, `g6q-train.json`)
  plus ~35 min of eval (3 repeats x 2 modes x ~5.5 min per 7-set Lightning pass), plus
  ~1.5 min per server start (91 s with the LoRA server, this revision). Training and eval
  share the laptop GPU, so they run one after the other, not overlapped.
- **LR is the dominant knob** [sourced, research §1.1, arXiv 2601.22708]. Sweep
  {5e-5, 1e-4, 2e-4} with 2 seeds per point, on the Tier 2 data: one seed per point cannot
  separate LR from seed noise. 6 runs is about 8.5 h with eval [arithmetic]. If an
  endpoint wins, extend one step past it (2.5e-5 or 4e-4) at 2 seeds before choosing,
  about 2.8 h more [inference; arithmetic]: a best value at the edge of the grid is not
  yet located.
- **Keep the rest fixed**: r=16, alpha=32, 8 records per step, 2 epochs [measured,
  `g6q-train.json`], and today's targets.
- **Today's targets** [code, `g6_train.py` TARGETS; measured, `g6q-train.json`]: attention
  q/k/v/o, the Mamba `in_proj` and the shared experts' up/down; 93 modules, 11.4M
  trainable parameters. The routed experts are **not** trained, so the research's
  "all linear layers" default [sourced, research §1.1] does not describe our runs.
- **Expert LoRA as one later arm** [inference]: `expert_lora.py` (per-expert or shared
  adapters inside the experts' forward) exists, but no G6 run used it. After the LR sweep,
  try it as one arm with 2 seeds if it fits 24 GB.
- **Run on Lightning first** (3B active): it is cheaper per run [inference].

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

## Exit criteria and decision tree [inference]

- **Tier 1 is done when**:
  - items load by hash and the cwd assert is in;
  - the scorer test file passes and the hand audit is reported per set, with every set
    at 18 of 20 or better (90%, a bar set before looking [inference]); a set below it
    has its scorer fixed before it carries a decision;
  - `decision_rule.py` (primary metric, paired CI, Holm, row flags) reports every
    comparison, and `pair_items.py` applies Holm to row wins;
  - the skeleton script reports max similarity to train for every dev item;
  - promql truth comes from the fixture Prometheus (1.6), and every result carries the
    cost metrics (1.7);
  - the locked test is sealed: manifest hash committed, ledger started.
- **Tier 2 is done when** the cleaned data passes the dedupe, cap and control checks, and
  one bundle candidate has been judged on dev against G6q at the same seeds (2.2).
- **A final candidate** is trained at 2 seeds. The seed to serve is named before the test
  is opened and is the only one ranked; the other seed must also qualify, so one lucky
  seed cannot carry it (`decision_rule.py --named SEED_A --partner SEED_B`). G6q (seed 0,
  the existing adapter) and G6u enter as the fixed adapters they are: the locked test
  measures exactly what would be served, and neither was chosen on it. The partner rule
  guards the candidate, whose recipe is chosen after many looks at dev.
- **On the locked test**, the arms are the base, G6q, G6u and the candidate's two seeds,
  scored in the mode that will be served (Needs David: is reasoning needed?). An arm
  **qualifies** when it scores above the base with a Holm-adjusted p below 0.05 across
  these comparisons with the base, and it has no row flag against the base (1.3). Then:
  1. the leader is the qualifying arm with the highest primary metric;
  2. another qualifying arm is **non-inferior** when its paired CI against the leader has
     a lower bound above -0.02 (the margin; Needs David). Among non-inferior arms, serve
     the cheapest whose completion tokens per item are at most 0.8x the leader's at the
     95% bootstrap bound (1.3); failing that, keep the served arm if it is non-inferior,
     since a tie is no reason to switch; failing that, the leader. Every qualifying arm is
     compared with the leader, not only the runner-up. These CIs are not Holm-adjusted, and
     the leader is the best of noisy scores, which favours it; both err toward keeping the
     leader. With CIs about 0.1 wide, a 0.02 margin is rarely met unless two arms are
     truly level, so in practice the rule serves the leader unless David widens it;
  3. if no arm qualifies, serve the base, and treat adapter work as unproven for this
     task mix.

  The test is not reopened for this candidate.
- **Change approach** if, on dev, any of the three G6q seeds (0, 1, 2) has a Holm-adjusted
  p >= 0.05 against the base, the family being the three seeds (`decision_rule.py` with
  the three seeds as its only arms, since its Holm family is the arms it is given): G6q's margin then depends
  on the seed. Stop ranking adapters on dev by point estimate, and enlarge the eval (1.1
  sizes) before training more. Judge the rocky_task flag apart: if 2 of the 3 seeds carry
  it, the loss belongs to G6q's data, not to one seed's noise, and G6q stays behind G6u.

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

### Judge pass 2 (2026-10-04, this revision)

Claims re-run on the files, not read from the audit:

| claim | verdict |
|---|---|
| 1,248 records, 157 exact duplicates, prompts at 10 and 12 copies, 6/60 and 4/48 | CONFIRMED |
| 0 of 2,424 assistant turns hold reasoning; no system or second user turns | CONFIRMED |
| promql in 58 records; 99,387 assistant tokens | CONFIRMED |
| 544 answer "Based on `x --help`" | CORRECTED: 544 say "Based on `<source>`"; 329 cite `--help` |
| training loss "reached 0.0016" | CORRECTED: single-step minimum; epoch-2 mean 0.17 |
| all scorer cases in 1.5 | CONFIRMED through the real `score()` |
| sign-test table, all 11 cells | CONFIRMED |
| G6q alert 7, 3, 5; 8B alert 78% to 11% | CONFIRMED |
| `g6_train.py` exits 0 after an abort | CONFIRMED (saves `-partial`, falls through) |
| unknown: does Lightning LoRA train the experts | RESOLVED: shared experts only, not routed |
| the six arXiv IDs | CONFIRMED: each exists and says what is attributed |

Design gaps fixed in this revision: the base anchor and G6q's rocky_task regression;
Holm making the regression gate toothless; item pooling handing decisions to the flag
sets; a locked test in a public repo with no opening ledger; seed-free LR comparisons;
the "all linear layers" mismatch; no scorer error rate; no cost metrics; no exit criteria.

### Judge pass 3 (2026-10-04, an independent judge session)

Each finding was checked on the files or by running `decision_rule.py`, not taken on the
judge's word:

| finding | verdict |
|---|---|
| a set-level regression margin misses G6q's rocky_task loss | CONFIRMED: rocky nets a gain of 15.7 item-rows; the rule is now per row (1.3) |
| G6u, not G6q, leads on the primary metric | CONFIRMED on dev; the gap's CI includes 0 (1.3 dry run) |
| the tree's branch 2 serves the base when G6q fails, even if G6u qualifies | CONFIRMED; the tree now ranks base, G6q, G6u and the candidate |
| the reseeds "need nothing from Tier 1" | REFUTED: SEED and LR are hard-coded (`g6_train.py:96`), and `gpurun.sh` does not pass them (3.2) |
| a 2-item margin fires on run-to-run noise | CONFIRMED: 12 of 396 single-run pairs (3.0%), ~0.67 per 22-row comparison; the range condition guards it and did not change the dry run (1.3) |
| "with 20-25 rows" (1.3) | CORRECTED: 22 rows on the 7 sets [measured] |
| `correct_where_scorable` read per item | FIXED: per item it is `correct`; read by name, the no_tool row was silently empty (1.3) |
| the rule ignores cost | FIXED: a tie goes to the arm with 20% fewer completion tokens per item; G6q costs 77 per item, G6u 381 (1.3); revised in pass 4 |
| "39 name a different flag and question" (1.4) | CORRECTED: 39 differ in flag or question, 23 in the flag |
| "seed-matched" is undefined (2.2) | FIXED: the same seeds, 0 and 1, otherwise identical |
| an LR that wins at the grid's edge is not located (3.2) | FIXED: extend one step past it |
| a one-seed candidate can pass the locked test by luck | FIXED: 2 seeds, the served one named first (Exit criteria) |
| the cost omits server start; eval and training could overlap (3.2) | FIXED: 91 s per start; they share the GPU, so they run in sequence |
| Tier 1's exit omits the fixture Prometheus and the cost metrics | FIXED |
| the primary metric's mode (thinking on or off) is unspecified | FIXED: the mode that will be served |
| alert's 9 items cannot carry a regression | KEPT in the metric; the locked test sizes alert to 30+ (1.1) |

### Judge pass 4 (2026-10-04, an independent judge session)

Each finding was checked on the code or by running `decision_rule.py`:

| finding | verdict |
|---|---|
| a set missing for one arm is dropped for all, silently; no set at all crashes | CONFIRMED; the script now refuses partial coverage (`--allow-partial` overrides and says so) and exits cleanly with no set |
| the two-seed rule (Exit criteria) is not in the code | CONFIRMED; `--named`/`--partner` implement it: the partner is never served, the named seed needs it to qualify |
| the cost tie-break decides most ties: CIs are ~0.1 wide, so "includes 0" is common, and a 9-item row drives the gap | CONFIRMED arithmetically; a cost win now needs non-inferiority (lower bound above -0.02) and 0.8x at the 95% token bound |
| only the top two arms are compared, and ties are not transitive | CONFIRMED; every qualifying arm is compared with the leader |
| "Change approach" fires on any seed's unadjusted CI | CONFIRMED; Holm over the 3 seeds, and the rocky_task flag judged apart |
| the served arm is not printed against the pick | CONFIRMED; printed whenever they differ |
| a 4-cluster row drops out of 2.4% of resamples | CORRECTED: 1.7% (34 of 2,000) on the script's resamples; no kind lost; the locked test needs 10+ clusters per row (1.3) |
| "blocked from G6q only by its rocky_task flag" reads backwards | FIXED (Needs David) |
| "repeats vary under 6%" | CORRECTED: 5.9% of the mean summed, up to 52% on one set; the rule uses the bootstrap bound |
| the scorer-agreement gate (1.5) has no bar and no consumer | FIXED: 90% per set, in Tier 1's exit |
| REPS="r1" gives a range of 0 | FIXED: the script refuses fewer than 2 repeats |
| make the cheaper arm also match the leader on trap and no_tool | NOT ADOPTED: row flags against the base and non-inferiority already guard it; per-kind matching adds unplanned comparisons |

The dry run is unchanged by the fixes: identical scores, CIs and Holm values, and it
serves G6u [measured].

### Judge pass 5 (2026-10-04, an independent judge session)

The judge re-ran the dry run, the refusals, the named/partner paths and the
non-inferiority path on a scratch copy, and re-measured the doc's numbers; all matched.
Its findings, each checked by running:

| finding | verdict |
|---|---|
| a turn with no `completion_tokens` counts as 0 tokens, so an arm served without usage data looks free and wins on cost | CONFIRMED in code (dev data is clean: 0 of 9,361 turns); the script now refuses |
| `--named X --partner X` silently serves the base | CONFIRMED; now an error, as is `--served` naming the partner |
| G6u is one seed while the candidate needs two | KEPT, explained (Exit criteria): G6q and G6u are fixed adapters measured as served; the partner rule guards a recipe chosen on dev |
| the non-inferiority CIs are unadjusted, and the leader is a noisy argmax | CONFIRMED; both favour the leader, now stated (step 2) |
| "6.0%" and "52%" use different bases | CONFIRMED: 6.0% was range over min; both are now range over mean (5.9%, 52%) |
| the Change-approach Holm family is whatever arms the script is given | FIXED: run it with the three seeds as the only arms |

### Why the passes stop here

Each pass found less, and of a smaller kind. Pass 3 found that the rule missed G6q's
rocky_task loss and never ranked G6u; with both fixed, the dry run's pick became G6u. Pass
4 found that cost decided most ties and that the two-seed rule existed only in prose; its
fixes changed no pick on today's data, but would have with G6q's flag cleared (G6q before,
G6u after). Pass 5 found two guards against malformed input (a missing token count, a seed
named twice) and four wording points; none changes a pick on well-formed data, and each was
fixed and re-run.

What remains is not a flaw in the plan that a further pass could fix:
- **David's decisions**: the non-inferiority margin, G6q or G6u until the locked test,
  whether reasoning is needed, the SEED/LR change (Needs David). A judge can show what each
  choice does, as 1.3 does for the margin, but not make it.
- **Data that does not exist yet**: the locked test, the reseeds and the scorer audit.
  Their results can reopen the plan; more reading of the plan cannot stand in for them.
- **Stated limits**: dev numbers are not test numbers, tokens are measured on the eval mix,
  and alert's 9 items cannot carry a decision. Each is labelled where it is used.

A further pass would judge wording. The next real test of the rule is its first run on
data it has not seen.

## Open unknowns

- Seed spread of a Lightning adapter (3.2: the reseeds answer it).
- Whether expert LoRA fits 24 GB with the lean scan (3.2).
- NF4-trained vs bf16-trained Qwen3.8 adapter quality (4.2).
- vLLM 0.29.0 batch invariance with hybrids plus LoRA (4.2).
- GRPO on a hybrid MoE in 24 GB (4.3).
- An open PromQL dataset.
- NEFTune replications.

## Needs David

- **Download go**, per dataset: When2Call first (4.1); any other row in the volume table.
- **Is reasoning needed?** If yes, train the mix (3.3). If no, serve thinking off and drop
  thinking-on as the primary mode. On dev, thinking off is not measurably worse for
  either adapter on the primary metric (G6u loses 4.67 items of v2 task), at 4.8x fewer
  tokens for G6u; the base needs thinking (1.3). The dev sets cannot show a reasoning
  need (1.7).
- **Who writes the locked test set** (1.1), and an agreement that its items are never read
  per item during data work.
- **G6q or G6u until the locked test.** L7/L8 chose G6q; the dev dry run of 1.3's rule
  picks G6u, because G6q carries a rocky_task flag. G6q writes 5.0x fewer tokens at a
  primary metric the dry run cannot separate (1.3, 1.8). With thinking off, the two
  cost the same and the rule again picks G6u, so G6q's cost case depends on serving
  thinking on. Both are dev numbers.
- **The non-inferiority margin** (default 0.02 on the primary metric). It is the score
  David will give up for a cheaper model. On dev, G6q would need a margin above 0.094
  even with its flag cleared (1.3).
- **The SEED/LR change** to `g6_train.py` and `gpurun.sh` (3.2), before any reseed.
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
3. The first GPU work, in parallel with 1 and 2: the G6q reseeds (3.2), judged on dev with
   3 repeats and the base anchor by `decision_rule.py`. They need the SEED/LR change in
   `g6_train.py` and `gpurun.sh` first, and nothing else from Tier 1.
4. Then, unless "Change approach" fired, the LR sweep on the Tier 2 data with a grouped
   validation split, 2 seeds per point (3.1, 3.2).
5. Write the locked test set in parallel. Use it only when a candidate is final.

# Harness audit: training, data and eval (G6q lineage)

Audited 2026-10-04, read-only, CPU only. Scope: `probes/g6_train.py`, `q2_train.py`,
`lean_lora.py`, `chunked_ce.py`, `attn_bf16.py`; builders `g6p/g6q/g6t/g6w_build.py`;
eval `g7a_eval.py`, `g6_eval.sh`, `g6_compare.py`, `pair_items.py`, `noise_eval.sh`,
`noise_summary.py`, and gpu-lab `bench/research_eval.py` (read-only). Data audited:
`results/research_dataset_g6q.json` (1,248 records) against the 478 eval items of the seven
sets. Scratch scripts and outputs: `/home/david/.claude/jobs/38f8859e/tmp/audit/`.

Labels: **measured** = I ran it and have the output; **code** = read from source, not run;
**inference** = reasoning from the above. No GPU, no download, nothing committed.

## The 10 fixes that matter, ranked

1. **Freeze a sealed test set that no data, recipe or selection decision has seen (HIGH).**
   All 478 items serve as design input, gate and selection. G6q's data was written after
   reading per-item failures on the judged items (`g6q_build.py:3-14`: "11-12 of the 18
   promql items", "per-item read, 2026-09-28"); ~10 adapters (G6..G6w) were then ranked on
   the same items. Write a new set after the data freeze, from different structures, and
   score it once per finalist.
2. **Check contamination on skeletons, not words (HIGH).** Six of the 9 alert items and
   about half of the 18 promql items have a structural twin in G6q's new records that
   differs only by metric name. The word,
   Jaccard and metric-name checks pass by construction. Mask metric names, durations and
   thresholds, compare operator skeletons, and either drop twins or report the affected
   rows as in-distribution (section 1.3).
3. **Stop quoting the gate run as the result (HIGH).** G6q alert, thinking on, over three
   identical runs is 7, 3, 5 of 9; promql 15, 13, 13 of 18. The mean is 5.0 and 13.7.
   `docs/plan.md` already records the means (N1 result, lines 1505-1513), but the summary
   table at `docs/lightning-training.md:128` still quotes the first, highest run
   ("promql 0.833 and alert 7/9 thinking on") with no correction. Fix that row; going
   forward, gate on the mean of at least 3 repeats; make the 4-item win/loss rule a sign
   test; enlarge `alert` (n=9) to 30+ items.
4. **Decide whether "thinking on" is meant to think (HIGH).** 0 of 2,424 assistant turns
   in G6q hold any reasoning text; every thinking-on turn is trained as `</think>` straight
   after the masked `<think>` opener. The adapter is a thinking-off model in both modes.
   No eval measures what that costs on reasoning tasks (`general` is 45 trivial items).
   Add a reasoning-required held-out set, or train with traces (G6u) and re-gate.
5. **Dedupe and cap templates (MED).** 157 records are exact duplicates; 1,091 unique
   message lists in 1,248; 6 web_research prompts appear 10 times each, 4 conversational
   prompts 12 times each; 544 records (43.6%) answer in "Based on `x --help`...". With 2
   epochs, 12 duplicate copies means 24 gradient passes over one answer.
6. **Harden the scorer where it passes wrong answers (MED).** Measured examples in
   section 8.2: a shotgun list of numbers passes `promql` number items; "No problem, use
   `--fake`" counts as a denial; `alert` accepts `for: 8m` and `> 71` when 5m and 85 were
   asked. Fix list in 8.2.
7. **Commit an item manifest and assert the cwd (MED).** Outside a git repo the v1 set is
   166 items, not 158, and 39 of the 154 shared ids point at different flags and questions.
   Write items to a hashed JSON; fail if cwd or host tool versions differ from it.
8. **Add validation to training (MED).** No held-out loss, no early stopping, no seed
   repeat; step loss reaches 0.0016; the per-epoch checkpoint (step 156) was never
   evaluated. Hold out 5-10% of records grouped by spec/tool and log validation loss every
   52 steps; train the final candidate with 2-3 seeds.
9. **Add the controls the new behaviors need (MED).** The `promql` tool is visible in only
   58 of 1,248 records, so no record shows it offered and not used. No record has a
   system message, a second user turn, or a thinking-off no-tool answer outside alert
   rules (11 records). Add these before deployment, where the triage agent will have them.
10. **Keep the data filter and the eval scorer apart (LOW-MED).** Training records were
    accepted only if the eval's own `grounded_token`, `GLOBAL_DENIAL` and `score_alert`
    passed them (and G6t traces by the eval scorer). The adapter can learn the scorer's
    quirks, then be graded by it. Use an independent checker for filtering, or audit by
    hand a sample of eval passes. Also pin the lab state the `vllm:` no-data items depend on.

---

## 1. Contamination

Method [measured, `contam*.py`]: eval prompts = the `question` of each item as the base runs
asked it (478 items, ids identical across 418 result files; Qwen yardstick runs use the same
ids and text). Normalised: lowercase, whitespace collapsed, whitespace tokens (a second
pass with punctuation stripped gave the same counts, +2 tool-result hits). Training text
split into user prompts, assistant targets (content plus tool-call arguments) and tool
results. n-gram sets of size 8 and 13.

### 1.1 n-gram counts per set [measured]

| set | items | items with a 13-gram (question long enough) | 13-gram hits | 8-gram hits in prompts | 8-gram in targets | 8-gram in tool results |
|---|---|---|---|---|---|---|
| v1 | 158 | 66 | 0 | 7 | 0 | 4 |
| v2 | 134 | 79 | 0 | 0 | 0 | 0 |
| rocky | 102 | 71 | 0 | 0 | 0 | 0 |
| promqlcat | 18 | 4 | 0 | 0 | 0 | 0 |
| general | 45 | 6 | 0 | 0 | 0 | 0 |
| alert | 9 | 9 | 0 | 0 | 0 | 0 |
| trap3 | 12 | 1 | 0 | 0 | 0 | 0 |

Worst examples: all seven v1 prompt hits are `seen_tool` items sharing the template
sentence with a training prompt about the same tool, e.g. `seen_tool-git_rebase-0` shares
"rebase. which command-line option in git rebase should" with training record 717;
`seen_tool-tar-0` and `-5` share "which command-line option in tar should i use?" with
records 331, 377, 662. The 4 tool-result hits are flag descriptions in help text that
training also showed (by design for `seen_tool`).

Read: literal overlap is near zero, so word-level leakage is not the risk. Short items
(25 of 45 `general` questions have fewer than 8 tokens) cannot hit an 8-gram at all, so
this test is weak for them; use 1.2 and 1.3.

### 1.2 Nearest-prompt similarity and flag overlap [measured]

Word-set Jaccard against the closest training prompt (`contam5.py`):

Max Jaccard / items >= 0.5: v1 0.73 / 45 (`seen_tool-git_rebase-2`, template + same tool),
v2 0.65 / 11 (`held_out2-nfsstat-0` vs "show output in tebibytes ... in free"), trap3 0.64 /
1, rocky 0.62 / 9, general 0.62 / 4 ("capital of Japan" vs "capital city of Malaysia"),
promqlcat 0.47 / 0 (`promql-temp-desktop` vs "What graphics clock is the desktop GPU
running at"), alert 0.34 / 0.

Most of the similarity is the shared phrasing template: the eval builder draws its
questions from the training generator's own templates (`research_eval.py`:
`CLI_PROMPT_TEMPLATES`, `TRAP_PROMPT_TEMPLATES`). So the held-out splits test unseen
tools, not unseen phrasing [code].

Other direct checks [measured]:
- fake (tool, flag) pairs: 51 in eval, 18 in training, overlap 0; fake flag names: overlap 0.
- eval tools that appear as a training `tool` field: git, grep, journalctl, tar, exactly the
  `seen_tool` split. Tools in v2, rocky and trap3 appear nowhere as commands (a word match
  on `diff`, `file`, `patch` etc. was all `git diff`, English prose or `docker cp`).
- `seen_tool` is meant to hold flags training never asked about, but the exclusion list is
  the old `research_dataset.json`, not v3/G6p/G6q [code: `training_inventory()`]. One of
  25 items (`seen_tool-git_rebase-1`, `--merge`) is asked about directly in G6q's data
  [measured]. All 25 flags appear in training tool output (by design).

### 1.3 Structural twins (the real exposure) [measured text, twin pairing is inference]

`g6q_build.py` excludes eval metric names and alertnames and requires Jaccard < 0.5
(max 0.467, `results/g6q-build.json`). It does not look at the shape of the question or of
the rule. Pairings I find by reading the 9 alert and 18 promql eval items against the 29
promql and 20 alert specs:

| eval item | structural twin in G6q (only the metric differs) |
|---|---|
| alert `GPUMetricsAbsent`: no series exists at all for 5m; checks 8m False / 22m True | `PrometheusSelfDown`: `absent(...)` for 5m; series `_x25`; the same checks 8m False / 22m True |
| alert `GPUMemoryNearlyFull`: used / total per node > 0.95 for 10m | `HostMemoryPressure`: used / total > 0.9 for 5m, per node |
| alert `BatteryLowOnBattery`: A < 20 and B is 0 on the same node, immediate | `GPUClockStuckLow` / `BatteryDrainFast`: A below x `and on(node)` B, same shape |
| alert `GPUThrottling`: metric is 1 for 3m | `GPUExporterFailing`: metric == 0 for 3m |
| alert `HighPowerDraw`: average over 5m above 300, immediate | `GPUMemBusSaturated` / `BatteryDrainFast`: `*_over_time[10m] > N` |
| alert `ScrapeFlapping`: changes in 10m | `CPUTempRising`: `delta(...[10m]) > 20` |
| promql `maxtemp-1h` / `util-laptop-10m` (`max_over_time[1h]`, `avg_over_time[10m]`) | "peak CPU core temperature in the past hour", "average core clock over the past quarter hour" |
| promql `mem-desktop-gib` (`/ 1024`), `ac` (yes/no on a 0/1 gauge) | "RAM in use, in GiB" (`/ 1024^3`), "Is the laptop battery discharging?" |
| promql `vllm-running/-kv/-waiting` (expect no data) | 4 records on `vllm:prompt_tokens_total`, `vllm:num_preemptions_total`, answered "empty result, the server is not scraped" |
| promql `up-count`, `down-jobs` | `topk(1, scrape_samples_scraped)`, `scrape_samples_scraped{job=...}` |

Evidence the records were aimed at the eval: the builder docstring names the eval's tool,
its metric catalog ("the same description the eval's promqlcat set builds") and the eval's
own alert scorer; the 40 alert records all use the eval's series syntax and the same
two-point check times. A model that learns "answer with the promql tool, then restate the
value" or "`absent()` ... for 5m" from these records is not showing generalisation to
unseen operator tasks. Severity high for the two rows the gate calls G6q's wins
(promql, alert). Note also the lab-state coupling: the eval's nodata items are right only
while the vLLM targets are down, and the training records teach the same fact [code].

## 2. Composition [measured, `comp.py`, `dups.py`]

1,248 records = 950 v3 + 200 task (G6p) + 98 (G6q). All records: roles user/assistant/tool
only (1,248 user, 2,424 assistant, 1,176 tool turns); no system message, no second user turn.

Types (n; unique prompts where fewer): cli_grounded 380, task_procedure 200 (G6p),
general_answer 100 (Dolly), compose 80, trap_refusal 72 (18 unique prompts), web_research 60
(**6** unique), promql_live 58 (29 specs), web_fallback 48, conversational_replay 48 (**4**
unique), alert_direct 40 (20 specs), man_fallback 36, lookup_failed 36, asserted_trap 36,
arithmetic 30, trap_lookup_failed 24 (12 unique).

- Tool vs no tool: 1,030 records call a tool (884 with 1 call, 146 with 2), 218 do not
  (100 Dolly, 48 replay, 40 alert, 30 arithmetic). Calls: bash 1,006, web_search 108,
  promql 62. Thinking off: 357 records, 346 of them with a tool call; 11 off records have no
  call (all alert). Thinking-off plain answers to ordinary questions were never trained.
- Duplicates: 157 exact duplicate records; 1,059 message lists occur once, 18 occur 4
  times, 10 occur 10 times (6 lists), 12 times (4 lists). 22 lists appear in both thinking
  modes. 120 prompts have more than one distinct continuation. Word-Jaccard >= 0.8 prompt
  pairs: 1,183 (web_research 270, conversational_replay 264, cli_grounded 160, trap_refusal
  108); 5-gram target pairs >= 0.8: 1,023. Same prompt, same answer, up to 12 times.
- Template: 544 of 1,248 (43.6%) have "Based on `x --help`" in the answer (cli_grounded 380,
  compose 80, web_fallback 48, man_fallback 36); the commonest 3-word openings are
  "Based on `git" (179), "Based on `docker" (95). `plan.md` already names this as the
  template collapse that costs rocky_task.
- Decline / unknown / ask (regex on the final answer): decline 93 (trap_refusal 72,
  lookup_failed 18, general 3), "could not verify / no data" 28 (trap_lookup_failed 24,
  promql 4): about 9.7% of records. Clarifying questions: **0**; no final answer ends in "?".
  A user who asks something ambiguous has no training example of being asked back.
- Length: prompt p50 69 chars (max 285); assistant text p50 197 chars (max 574); tool
  result p50 1,188 chars (max 7,647). Tokens per record p50 860, p90 1,503, p99 1,867, max
  1,892; trained tokens 99,387 of 1,178,797 (8.4%). Truncation at 2,048: 0 records (the
  v3 data at 1,024 lost 439 of 950 tails, including their final answers: G6/G6r/G6p
  trained on cut targets; `G6p2048` is the right control and exists).
- Controls: the weaknesses targeted by the last two steps are task procedures (200),
  promql (58) and alert rules (40) = 298 records. The other 950 are v3, which act as
  controls for flag lookup and denial, but there are **no controls for the new tools**:
  `promql` is in the tool list of 58 records only; no record offers `promql` and answers
  with `bash` or no tool.

## 3. Empty think [measured]

- Of 2,424 assistant turns, **0** contain `<think>`, `</think>` or `reasoning_content` text;
  0 records carry a base trace (G6u adds traces; G6q does not).
- In the run (RENDER=think): 1,676 default-mode turns train `</think>` right after the
  masked `<think>\n` opener; 748 turns in the 357 off records have the empty block masked.
  Every target has an empty think block; **none has reasoning**. `plan.md` agrees (915 of
  919 thinking-on eval turns have empty reasoning).
- Consequence [inference]: "thinking on, 4,096 tokens" is a thinking-off run with a larger
  budget (4 of 919 turns never close `</think>`), so the gate's primary mode cannot support
  any claim about reasoning, and the 45-item `general` set (e.g. "What is 3 cubed?")
  cannot show a reasoning regression. The Q2 trainer renders the same way [code, `q38.py`].

## 4. Loss masking [code, confirmed by measurement]

Code (`g6_train.py:encode`, `q38.Render.encode`): labels start at -100; for each
`<|im_start|>assistant\n` header, the span from the first token after the (masked) think
opener through the first `<|im_end|>` inclusive is copied into labels. Headers, openers,
system/user turns and tool results stay masked.

Measured (CPU, training-base tokenizer from the local cache, the `DRY_RUN` path exec'd with
GPU modules stubbed, `maskcheck.py`): all 1,248 records at MAX_LEN 2,048 reproduce the
run's own numbers (99,387 trained of 1,178,797 tokens; parity 2,424/2,424). Walking every
token with the role set at its `<|im_start|>`: all 99,387 trained tokens are in assistant
turns; none in user turns (tool responses sit in `user` turns in this template), none is
`<|im_start|>`; 0 records have trained text containing `<tool_response>` or the tool
preamble. Renders: `.../audit/example_promql.txt`, `example_trap.txt`. **Masking is correct.**

Loss [code]: per-record token mean (`chunked_ce`, -100 ignored, first position shifted out)
times `n_items/n_step` = a token mean over the 8-record step, the Trainer's
`num_items_in_batch` objective; `chunked_ce.calls` 2,496 = 312 x 8 [measured].
`lean_lora.py` is not imported by `g6_train.py` or `q2_train.py` (and falls back whenever
dropout is not Identity, here 0.05): dead code on the candidate path. Low.

## 5. Template parity [measured and code]

- Chat template, tokenizer and special-tokens files are byte-identical between the training
  checkpoint and the served quantized checkpoint (`cmp`); only `generation_config.json`
  differs, and the eval sets temperature 0 explicitly. Tool-call XML is the template's own;
  training turns JSON-string arguments into objects as vLLM does; `g7a_eval.py` parses that
  XML. [measured]
- Think tags: the thinking-on prompt ends in an open `<think>\n`; training masks that and
  trains `</think>` + content. Known gap: history turns read `<think>\n</think>` in training
  and `<think></think>` in the server prompt (785 of 2,424 turns, 886 tokens), design A;
  design B lost.
- The parity guard builds the "server prompt" with the local HF template, not vLLM's
  `/tokenize` [code]. For G6u it also matched vLLM's prompt token counts on 502 of 502
  turns [measured, `results/g6u-train.json`], which supports the HF rendering; G6q has no
  such check. The promql tool's metric catalog is captured from live Prometheus at build
  time and again at eval time, so a changed series set would change the prompt. Low.

## 6. Seeds and determinism

- Training seed 0: `torch.manual_seed(0)` after load, before `get_peft_model` (LoRA init and
  dropout RNG); per-epoch order from `torch.Generator().manual_seed(seed + epoch)` [code].
  No `use_deterministic_algorithms`; paged 8-bit AdamW and atomics: not bitwise
  reproducible [inference].
- **Training-seed variance is unmeasured**: every adapter is one seed-0 run [no second-seed
  training in `plan.md`]. G6q vs G6u vs G6t differ in data and in trajectory; a gap below
  seed variance is unattributed.
- Builder seeds: G6p 20260928, G6q and G6w 20260929 (opener/thinking assignment). G6q's
  records embed live Prometheus values, the metric catalog and promtool results; the
  committed JSON is the artefact, a rebuild would differ [code].
- Eval: `--seed 20260923` only picks items; `--sample-seed` is unused at temperature 0.
  Repeats use identical settings at concurrency 16.
- Run-to-run spread at temperature 0 [measured, `noise_summary.py`, G6q thinking on, 3
  runs]: held_out2 68/69/70 of 70; task 14/15/15; rocky_task 9/7/8; promql 15/13/13;
  **alert 7/3/5** (5 of 9 items stable); trap3 5/5/5. Repeats exist for five sets only
  (v2, rocky, promqlcat, alert, trap3). **v1 (held_out, the gate's must-win row; also trap,
  trap_control, no_tool) and general have a single thinking-on run per adapter.**

## 7. Validation [measured, `results/g6q-train.json`]

- No validation split, held-out loss or early stopping in `g6_train.py` or `q2_train.py`
  [code]. The only signal is training loss: 1.713 first step, mean 0.474 in epoch 1, 0.170
  in epoch 2, 0.074 last step, minimum single step 0.0016, 18 of the last 50 steps below 0.1.
  Epoch 2 revisits the same records (12.6% exact duplicates), so low loss is template
  memorisation, not a held-out signal.
- The per-epoch checkpoint (step 156) is written but never evaluated, so there is no
  loss-versus-eval curve; selection uses the same eval items (1.3, 8.5), so overfitting to
  the eval would look like progress. `q2_train.py` adds checkpoints every 52 steps; still
  no validation.

## 8. Eval robustness

### 8.1 Item counts [measured]

v1 158 (held_out 90, seen_tool 25, no_tool 20, trap 15, trap_control 8); v2 134 (held_out2
70, task 20, trap2 12, trap2_control 12, fix_cmd 10, two_flag 10); rocky 102 (rocky_held_out
56, rocky_task 20, rocky_trap_control 14, rocky_trap 12); promqlcat 18; general 45; alert 9;
trap3 12.

The gate's "must not lose" rows: trap 15, trap_control 8, no_tool 20, task 20, rocky_task
20, alert 9, promql 18. The rule counts a loss only when the adapter is worse by more than
4 items (`g6_compare.py:KIND/verdict`): for `alert` that is 5 of 9, and G6q's own
run-to-run range there is 4 [measured]. The gate can neither see a 4-item regression nor
distinguish adapters on these rows. Ceiling: held_out2 69/70, rocky_held_out 55/56,
fix_cmd 10/10, trap2 11/12 for G6q: three of the splits cannot discriminate.

### 8.2 Scorer brittleness [measured: `scorer_probe.py` calling `research_eval.score` with
synthetic answers; alert cases through the local promtool image, no pull]

Passes a wrong answer:
- `promql` number items accept any number in the answer within tolerance of any truth
  value: "It might be 20, 40, 52 or 55 degrees; I have not looked." with truth 55 and no
  tool call scores `correct: True` (`invented_value` flags it separately but the headline
  is `correct`).
- `promql` nodata: "The vLLM server is running 4 requests right now and the queue is
  empty." scores `correct: True` because the word "empty" matches `NODATA`.
- Trap `denied_heuristic` opens on yes/no: "No problem! Use `cp --parallel-copy -r src dst`"
  scores `denied: True`, `grounded: True`. 
- `flag` hit has no negation handling: "Do not use -r for that; there is no such option in
  cp." scores `hit_and_grounded: True` for a `-r` item.
- `general` regexes match restated question numbers: "144 divided by 12 is 13." passes
  `\b12\b`; "81 divided by 9 is 8." passes `\b9\b`; "The smallest prime is 1 (2 of 3...)"
  passes `\b(2|two)\b`; "212 degrees (not 100)" passes the boiling-point item.
- `alert` checks only two evaluation times: for GPUHot (asked >85 for 5m) a rule with
  `for: 8m` and rules with `> 71` or `> 89` all score correct; TargetDown (asked
  2m) rejects 1m and 5m. Duration and threshold are checked loosely, the shape strictly.
- `trap3` `noticed` is also true if the real tool's name appears anywhere in the answer
  (`has_token(real, final)`), e.g. an answer fabricating `--dry-run` for `rsnyc` that
  mentions `rsync` once.

Fails a right answer: "Six hundred thirteen." fails `\b613\b`; "7/8 is 87.5%." fails
`0?\.875`; "Yes, cp does not have a --parallel-copy option." scores `denied: False` (the
opening yes/no wins over the content); `instruction_ok` fails "A cat is a small mammal,
e.g. a tabby. It purrs." for `sentences == 2` and "It is blue." for `lower_with`. [code] A
`promql` answer in other units (GiB when MiB was asked) fails, and a truth equal to a number
in the question (e.g. "10" in "last 10 minutes") cannot score.

Other [code]: a call to a tool outside `TOOL_NAMES` is read as a plain answer; an answer
with no `</think>` is scored "truncated in think" even if it ended normally (`g7a_eval.py`),
so a format miss is an eval failure; `rules_from_answer` takes the first YAML block that
parses; live truth is read after the answer, so `promql` rows carry Prometheus drift on top
of model noise.

### 8.3 cwd, host and item identity [measured]

`research_eval.py` runs `<tool> --help` and `man` in the caller's cwd on the host's
binaries. `git diff -h` differs outside a git repo, so the same command and seed yield 158
items in the repo and **166 outside** (8 extra `seen_tool`); of the 154 ids the lists share,
**39 point at a different flag and question** (`seen_tool-git_rebase-0` is `--interactive`
in one, `--merge` in the other). Ids are `{split}-{tool}-{n}` and the n-th pick depends on
the RNG stream, so an id does not name an item across cwds or hosts. All 418 result files
here are 158/134/102/18/45/9/12 and the Qwen yardstick files hold identical ids and text
[verified], so existing comparisons stand. The risk is a future run from another directory
or host: `g6_eval.sh` and `noise_eval.sh` never `cd`, and `pair_items.py` joins by id and
would pair different items. The rocky set also depends on the docker image, `promql` on the
live lab (the vLLM no-data items hold only while those targets are down), `alert` on the
promtool image.

### 8.4 Repeats and statistics

- The gate (`g6_compare.py`) uses one run and a fixed 4-item band; repeats came later and
  cover five sets (6).
- `pair_items.py`: exact two-sided sign test on discordant items (item = mean over a side's
  repeats, so 0.33 vs 0 counts). It marks raw p < 0.05 with "<" and has **no Holm**; the
  docs say Holm is reported beside it, but it is computed only in `s1_compare.py`,
  `o70_compare.py`, `q2_compare.py`. With 20-25 rows per comparison one raw "<" is expected
  under the null. Items come in clusters of 6 per tool with shared templates and three
  repeats of one item are pseudo-replicates, yet the test treats them as independent
  [inference].
- `g6_compare.py`: `items = round((va - vb) * n * sign)`; `correct_where_scorable` is counted
  over n, not the scorable count (its header says so); no `__main__` guard (importing runs
  the report, hence `noise_summary.py` copies `headline()`). Low.

### 8.5 Design/judge overlap

- G6p was built because G6/G6r lost task and rocky_task on the eval; G6q because "G6p and
  G6P2048 reach for bash on 11-12 of the 18 promql items" and both adapters spent the
  3-call limit on `man prometheus-alerting-rules` on the 9 alert items (`g6p_build.py`,
  `g6q_build.py` docstrings). The data author read per-item behavior on the judged items.
- Contamination checks exclude metrics, flag sets and Jaccard >= 0.5 against all 478
  prompts; not structure (1.3).
- Training records pass the eval's own scorer functions (`grounded_token`, `GLOBAL_DENIAL`,
  `score_alert` for all 20 alert specs; accepted traces in G6t/G6u). `plan.md` notes a
  correct G6q alert whose text claims a "promtool rule --help" lookup no call made: the
  scorer does not check it.
- About 10 adapters, 14 runs each, were ranked on the same items; "first to pass" and "the
  only adapter that beats the yardstick in both modes" are selections among many on one
  test set [inference]; quoted margins carry a winner's-curse bias (fix 3).

## 9. Smaller findings

- `attn_bf16.py` runs `g6_train.py` through `runpy` after patching `route1.load`; the run
  recorded `attn_linear4bit 0`, `attn_bf16_linear 24`, `lm_head` bf16, 93 LoRA modules,
  11,359,232 trainable parameters [measured, `g6q-train.json`]. Correct.
- `g6_train.py` saves `-partial` on a guard abort and still exits 0 (`q2_train.py` exits 3
  on OOM): a wrapper that checks only the exit status reads an abort as success. Low.
- web_search succeeds in 60 training records but is "unavailable" in the eval and the lab
  (`g6w_build.py` found this; G6w was dropped, G6q keeps it). The adapter is taught that a
  search returns an answer. Medium for deployment, none for the current eval.

## 10. Not verified

- The twin pairing in 1.3 is my reading of eval prompts and training specs; a mechanical
  skeleton comparison is the first step of fix 2.
- Kernel numerics and the 4-bit expert path (out of scope). No eval was rerun.

Scratch files: `/home/david/.claude/jobs/38f8859e/tmp/audit/` (`contam*.py`, `comp.py`,
`dups.py`, `maskcheck.py` run via `docker run gpu-lab:training` without `--gpus`,
`scorer_probe.py`, `list_in_repo.txt`/`list_out_repo.txt`, `n1.json`).

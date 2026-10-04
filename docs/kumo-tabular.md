# Kumo Tabular, steelman second pass (2026-10-04)

Why this file exists: the verdict lives in session memory; this is the evidence and the
only copy of the conditional trial (section 4). Nothing was downloaded or run on a GPU.

Labels: [M]=measured here, [S]=sourced (URL), [A]=arithmetic, [U]=unknown.
Scripts were scratch (`mat.py`, `router.py`, `irt.py`, pure python, not kept): the host python has no numpy or sklearn [M]; use the `gpu-lab:training` image for anything heavier. Read-only on repo data.

## Verdict
Not worth a download now. Nothing in the lab clears the bar over a numpy-class baseline. The prior "not big enough" argument WAS backwards in framing: the lab's tables are in the right size range for in-context models (hundreds of rows). The binding constraints are signal and labels, and they fail there. The one candidate with enough rows (router, 490 items) has weak, single-run labels and almost no features. Re-open only after the router study produces repeats (trial spec at the end). Update: the repeats exist and the gate passes, but cheap routers still gain nothing (section 4, Result).

## 1. Why NVIDIA released it
- [S] Model card https://huggingface.co/nvidia/Kumo-Tabular: classification and regression on tabular data; labeled rows as context, predict new rows in one forward pass; weights OpenMDW 1.1; runs via structured-data-models.
- [S] https://github.com/NVIDIA/structured-data-models : GPU-native library for tabular, relational, time-series practitioners; install from git main (PyTorch >= 2.7, Python 3.11+); also ships a KumoRelational variant.
- [S] Coverage (unite.ai, aireiter review): released 2026-09-29; 28M-215M params; pretrained on synthetic plus public data; target tasks churn, fraud, default, demand, price (replace per-question GBT pipelines); NVIDIA reports Elo 1950, 26x faster than LimiX-2 on one RTX 6000 Pro. These are vendor numbers [U independently].
- [S] Kumo AI acquisition ~USD 400M, founders joined NVIDIA May 2026 (pulse2.com, barchart); KumoRFM-2 (relational, April 2026) is the sibling product. So: enterprise-data-warehouse play, a GPU-demand and ecosystem move; not aimed at LLM training/serving. Row/col/class limits (60k/100/10) are from the prior memory, not re-verified on the pages I fetched [U].

## 2. Re-judging on signal and labels
In-context tabular FMs (TabPFN family, https://priorlabs.ai, TabPFN-2.5 up to 50k rows) suit small tables, so row count is not the obstacle. They help when there are informative numeric/categorical features, a few hundred to thousands of labeled rows, and an effect a linear model cannot catch. The lab tables fail on features and on label reliability (below), not on size.

## 3. Candidates (all counts [M] unless stated)
Item matrix: 490 headline rows x 16 runs (a row is one eval item under one headline metric, keyed (set, split, metric, id) via `probes/noise_summary.py:headline`; some splits score two metrics, so 490 rows is not 490 distinct items: the eval has 478) (Qwen3.8 INT4 nothink x1, think-4k x1, low x4; Nano, base Lightning, G6q arms), all 490 common.

(a) Qwen3.8 think/no-think router. 490 items; labels from low1 vs nothink: think wins 87, nothink wins 55, tie 348. Oracle +55 reproduced (439 vs 384 best fixed arm). Replay: the 87 think-labelled items stay think>=nothink in 79-80/87 on each of low2-4, 0 reversals; the 55 nothink-labelled stay in 37-40/55. BUT nothink has ONE run, so its side of every label is unreplicated (the gate's 3-repeat rule fails); think low self-discordance is 38-50 items per run pair. Baselines (labels from low1, scored on low2-4 mean, 5-fold x10 random): always-nothink 352, always-think 391.7, split rule 403.4, LR (split+length) 402.8, 5-NN (word Jaccard) 398.4, oracle 440.7. Leave-one-split-out: split rule 391.7, LR 391.7, 5-NN 392.7 = no gain over always-think. So the easy +11 is a per-split memorisation that does not transfer. Kumo adds: possibly non-linear interactions among features, but the only features are split id and question text; text would need embeddings the lab does not compute. Plausible add: small. Fails gate until nothink repeats exist.

(b) Cheaper evals. 16 runs. Predict a held-out run's total (out of 490) from m items: subset mean RMSE 46.3/31.9/21.3 items at m=20/40/80; difficulty-calibrated regression on the other runs 41.9/28.8/19.3 (~10% better). Flips: 41-50 items of 490 flip between two same-config think runs, 46/50 of them mid-difficulty. Noise floor of a run (~±20 items) is the same size as the m=80 error, and the full suite costs ~26 min (1,558 s desktop, 2,194 s laptop [M]). Kumo adds: a better item-outcome imputer than OLS on difficulty, but with 16 rows (runs) as context, which is below useful in-context scale. Little.

(c) Training-record quality. 1,248 records in G6q (docs/lightning-training.md). Per-record quality labels: none stored; only G6t's 131 scorer-accepted reasoning records (rejected count not in docs [U]) and ~10 dataset-level ablation outcomes. Nothing to fit. Kumo adds: nothing without labels.

(d) KV room / OOM / peak memory. About 8 of 443 result JSONs carry peak fields [M]; prior memory counts ~33 probes, 120 launches (not re-counted [U]). Arithmetic already predicts the laptop-vs-desktop KV gap: other GPU clients 0.60 + card size delta ~0.10 = ~0.70 GiB vs 0.64 measured [A, from next-model-plan.md]. A learned model cannot beat a formula that is within 0.06 GiB at n=2 configs.

(e) Run-outcome / hyperparameter surrogate. ~11 training runs plus ~10 dataset variants, many knobs changed at once, noisy outcomes (G6q null 11-14 discordant items). Under ~20 correlated rows; surrogate would extrapolate. No.

(f) Prometheus anomaly detection. Desktop Prometheus [M, via ssh llm]: 1,185 series, head window ~2.7 h (1791115200-1791124846 s). No incident labels; no alert rules deployed (GPU-Lab branch `lab-alerts` has 7 rules and a canary, unmerged). Kumo is supervised; anomalies here are 4 incident types seen once or twice. Alert rules plus a canary is the fit. No.

(g) Published use of tabular FMs in LLM training/serving pipelines. Searched twice: found only general TabPFN material and arXiv 2609.18130 (tabular FMs as surrogates in evolutionary optimization). Found none inside LLM training/serving pipelines; search was shallow [U], so "none exist" is NOT claimed.

## 4. The only trial worth pre-registering (conditional): (a) router on Qwen3.8
Do not run until: nothink has >= 3 repeats on the same server config and >= 10 replicated think-wins items exist. Then:
- Download: not published on pages I read [U]; 215M params at bf16 = ~0.43 GB [A], plus git install of structured-data-models (PyTorch >= 2.7; cuDF optional). Needs David's go.
- Pin: the merge commit of PR #1037 or later (resolve with `git ls-remote`; hash not looked up, [U]). Large model is the default; set size explicitly.
- Data: all item rows; label = think-better on majority of repeats (nothink and think), ties dropped. Features: split, question length, tool-count/size features, plus a frozen cheap text embedding only if one already exists locally.
- Held-out protocol: leave-one-split-out AND a fresh repeat set generated after labels are frozen (labels from runs 1-2, scored on runs 3+).
- Success rule (set before the run): Kumo router beats the BEST of {always-think, split rule, numpy LR, 5-NN} by >= 8 items of ~490 on leave-one-split-out and on the fresh repeat, and its discordant items vs that baseline favour Kumo at sign p < 0.05 on each. Otherwise stays not adopted. Today the three cheap baselines add 0 items leave-one-split-out, so the bar for Kumo is "find signal LR cannot", for which no evidence exists.

**Result, 2026-10-04** [M, `probes/router_study.py`, protocol in its docstring, fixed before repeat 3 existed]: the precondition is met. Nothink has 3 repeats on the same server config (totals 352, 347, 346; think-low 384, 386, 391). Labels from repeats 1-2, replayed on repeat 3: 80 replicated think wins (gate >= 10: PASSES), 0 of 99 think labels reversed; 67 items are think wins in all 3 pairs. A per-item label router, an upper bound since it needs a label per item, gets 430 vs always-think 391 (+39; discordant 40-1). Cheap routers on the listed features (log question length, kind, metric direction), leave-one-split-out on repeat 3: set rule -8 (discordant 11-19, sign p 0.20), LR +4 (23-19, p 0.64), 5-NN -20 (23-43, p 0.019). `--with-correct` (545 rows, adds no_tool/general correctness; a sensitivity, not the gate) gives the same verdict (446 vs 401, label router +39). So the signal is per item and today's features do not carry it. The trial is now allowed by its own gate; Kumo's bar is >= 8 items over LR's 395 on these features or a local embedding, and the download still needs David's go.

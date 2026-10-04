# LoRA/QLoRA training, data and eval practices: survey mapped to our setup

Date: 2026-10-04. Read-only web research; nothing downloaded, nothing run.
Label key: **[S]** sourced (URL given; some from search-result summaries rather than a full read of the paper, flagged "summary"), **[I]** inference by the author of this note, **[U]** unknown / not found, **[A]** arithmetic. Nothing here is a measurement on our hardware.

Our setup in one line: QLoRA (NF4, r=16) on Nemotron 3.5 Lightning (hybrid Mamba/attention MoE, ~30B total / 3B active) and Qwen3.8-27B (hybrid Gated-DeltaNet/attention, think/no-think); a few thousand mostly synthetic examples with EMPTY think blocks; 478 eval items in 7 sets (9 to 158 per set); one 24 GB training GPU.

---

## 1. LoRA/QLoRA hyperparameters

### 1.1 Consensus findings

- **Learning rate is the dominant knob.** A 2026 unified study of LoRA variants found LoRA and variants are "pronounced[ly] sensitive" to learning rate versus other hyperparameters, and that properly tuned vanilla LoRA matches or beats most variants. [S] https://arxiv.org/abs/2601.22708 (abstract read).
- **LoRA LR is about 10x full-FT LR** (about 15x for runs of ~100 steps or fewer); optimal LR is roughly independent of rank because of the 1/r scaling. [S] https://thinkingmachines.ai/blog/lora/ (read; "LoRA Without Regret", 2025-09-29).
- **Apply LoRA to every weight matrix, including MLP and MoE layers.** Attention-only LoRA underperforms even when parameter count is matched (attention-only r=256 lost to MLP-only r=128). [S] same Thinking Machines post. Biderman et al. likewise found target-module choice matters more than rank. [S, summary] https://arxiv.org/abs/2405.09673 ("LoRA Learns Less and Forgets Less", 2024-05).
- **Rank only matters when capacity runs out.** LoRA tracks full FT until the adapter runs out of capacity for the dataset; degradation is graceful. Datasets cost roughly 1 bit (0.69 nats) per token to memorize; RL needs far less (rank 1 suffices for policy-gradient RL). [S] Thinking Machines post.
- **Large batch hurts LoRA more than full FT**, and raising rank does not fix it. [S] Thinking Machines post. Keep batch small (effective batch 8-32 sequences is within what we can afford anyway). [I]
- **alpha = 32 fixed** is the PEFT-standard the post used; standard init (A uniform, B zero) was optimal there. [S] Thinking Machines post. Unsloth advises alpha = r or 2r, 1-3 epochs, LR 2e-4 start, dropout 0.1-0.2 optional. [S] https://docs.unsloth.ai/get-started/fine-tuning-guide/lora-hyperparameters-guide (summary).
- **LoRA learns less, forgets less.** Full FT beat LoRA on code/math continued pretraining and instruction tuning, but LoRA kept base performance better on out-of-domain tasks and regularizes more. [S, summary] Biderman et al. URL above. Mapping: for a narrow ops assistant, "forgets less" is what we want; "learns less" is why facts do not go in (see 1.5).

### 1.2 Variants: rsLoRA, DoRA, LoRA+, PiSSA, OLoRA

| Variant | Claim | Status for us |
|---|---|---|
| rsLoRA (scale alpha/sqrt(r)) | Fixes slowed learning at high rank. [S] https://arxiv.org/abs/2312.03732 (2023-12, older than 2024: possibly outdated framing). | Irrelevant at r=16; only matters if we go to r>=64. Note Thinking Machines argues 1/r scaling already makes LR rank-independent, so rsLoRA and "rank-independent LR" are different conventions; do not mix them in one sweep. [I] |
| DoRA | Magnitude/direction split; secondary sources claim "closes half the gap" at 5-10% more VRAM. [S, weak: blog summary, not primary]. | Unverified for QLoRA on hybrid/MoE; vLLM LoRA serving support for DoRA is [U] (not searched). Skip until needed. |
| LoRA+ | Different LR for A and B. Unified study: generally strongest variant in its tables. [S, summary] https://arxiv.org/pdf/2601.22708 | Cheapest variant to try (one extra LR ratio); but only after the LR sweep. [I] |
| PiSSA / OLoRA | SVD / QR init for faster convergence; PiSSA +5 pts GSM8K on Mistral-7B vs LoRA in its own paper. [S, summary] https://arxiv.org/pdf/2406.01775 (OLoRA) and the search summary. Another study found regularized plain LoRA matches PiSSA. | PiSSA modifies the base weights (residual), which interacts badly with NF4 quantization and with serving the adapter separately in vLLM. [I, unverified]. Low priority. |

Net: variants give small or inconsistent gains relative to getting LR, target modules and data right. [S for "LR matters most": 2601.22708; I for application]

### 1.3 Target modules, hybrid and MoE specifics

- Our rank-16-all-linear should include attention q/k/v/o, MLP gate/up/down, and for Mamba layers the in/out projections. [I]
- **MoE experts**: Thinking Machines recommend a separate LoRA per expert with rank scaled by active experts so the LoRA-to-full-FT parameter ratio stays constant. [S] Thinking Machines post. PEFT 0.18+ has `target_parameters` that wraps fused 3D expert tensors (e.g. Qwen3MoeExperts `gate_up_proj`/`down_proj`). [S, summary] search result friendli.ai / PEFT docs; I did not open the PEFT docs, so verify. Whether our Lightning training path adapts experts or only attention + shared layers is [U] from this research (I was told not to read repo code). If experts are skipped, the post's finding predicts a capacity penalty. [I]
- **Router**: Unsloth disables router fine-tuning by default for stability on Qwen3.5 MoE. [S] https://unsloth.ai/docs/models/qwen3.5/fine-tune
- **QLoRA on hybrid/MoE**: Unsloth states "not recommended to do QLoRA (4-bit) training on the Qwen3.5 models" because of larger quantization differences, and says use bf16 LoRA; it lists 56 GB for 27B bf16 LoRA. [S] same URL. Qwen3.8-27B is a newer sibling (Gated DeltaNet + gated attention hybrid per its listing, https://friendli.ai/models/Qwen/Qwen3.8-27B-FP8), so the caveat plausibly applies. [I] That means our Qwen3.8-27B NF4 run is outside the vendor's recommended regime; bf16 does not fit on 24 GB. Cheap check: compare NF4-trained adapter served over the NF4 base vs served over bf16/FP8 base. [I]

### 1.4 Learning rate, epochs, NEFTune

- Start LoRA LR at ~10x the full-FT LR for the model size; Unsloth's 2e-4 is in the same neighbourhood for small models. For a 27B model, 2e-4 is likely high; sweep {5e-5, 1e-4, 2e-4} with one seed each at small scale, then confirm the winner with repeats. [I, based on S above]
- **Epochs**: 1-3; Unsloth says >3 usually not optimal. [S] Unsloth guide. A few thousand templated synthetic examples at 3+ epochs is the setup in which a loss of 0.0003 appears (memorization). [I]
- **Loss near zero is a smell, not a win.** On our data it reflected near-duplicate synthetic targets plus easy template tokens. [I; consistent with the Thinking Machines capacity framing: loss is bits per token, so near-0 means the data carries almost no information the model lacks.]
- **NEFTune**: ICLR 2024 paper reports AlpacaEval jumps (29.8% to 64.7% on LLaMA-2-7B with Alpaca). [S] https://arxiv.org/abs/2310.05914. Its gains were measured on chat-style win rates judged by GPT-4, a metric vulnerable to length bias. I found no replication or refutation in my searches ([U]). It adds noise to embeddings, which interacts unpredictably with exact-format tool-call outputs. Verdict: do not add; no evidence it helps structured output. [I]

### 1.5 The "facts did not inject" finding is expected

- Gekhman et al. (EMNLP 2024): fine-tuning examples with new knowledge are learned slower than consistent ones, and once learned they linearly increase hallucination; models mostly acquire facts in pretraining, fine-tuning teaches how to use them. [S] https://aclanthology.org/2024.emnlp-main.444 . Our result (confabulation) matches this. Retrieval is the right fix; training should teach "use retrieved/tool output, cite it, decline otherwise". [I]
- A 2026 ICML paper maps LoRA's limits as parametric memory (capacity, scaling). [S] https://arxiv.org/abs/2603.01097 (abstract only; I did not read results).

---

## 2. Training data

### 2.1 Quality vs quantity, diversity

- LIMA: 1,000 curated examples gave a 65B model competitive alignment; "almost all knowledge is learned in pretraining". [S] https://proceedings.neurips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract.html (2023, older than 2024: framing still cited, but it is about style alignment, not tool precision).
- 2024 follow-ups: quality-diversity trade-off exists, and increasing diversity improves worst-case instruction following. [S, summary] https://arxiv.org/pdf/2409.11378 and https://arxiv.org/pdf/2402.18191 ; diversity measurement and subset selection: https://arxiv.org/pdf/2402.02318 , https://arxiv.org/pdf/2502.17184.
- Mapping: a few thousand templated synthetic items is a diversity-poor regime. Measure it (distinct tool-name x flag x phrasing clusters; embedding-cluster counts) and cap items per template cluster. [I]

### 2.2 Dedup and decontamination

- Standard recipe: exact dedup, then MinHash-LSH near-dup (char n-gram), optionally embedding-based (SemDeDup: MinHash, then k-means clusters, prune within-cluster neighbours under a distance threshold). [S] https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/index.html (summary).
- **n-gram decontamination is insufficient**: paraphrased or translated test items bypass n-gram checks; a 13B model overfit to rephrased benchmark items reached GPT-4 level. Embedding search plus an LLM judge ("LLM Decontaminator") catches more. [S] https://arxiv.org/pdf/2311.04850 (2023-11, possibly dated but the mechanism is unchanged).
- **Our risk is sharper than the paper's**: our train data is synthetic and our evals are in-house; if the same generator/templates/prompter wrote both, near-paraphrase leakage is likely even with zero string overlap. [I] Do: (a) embed every train and eval item, flag nearest-neighbour cosine above a threshold chosen by hand-audit; (b) hold out whole templates/generators, not just rows, for eval; (c) report per-eval-item "max similarity to train".
- Synthetic-data targets also carry the generator's habits (see v1-v3 "templates collapse" in our own history, which I was told not to read; stated here only as inference).

### 2.3 Synthetic generation

- **Magpie** (ICLR 2025): prompt an aligned LLM with only the template prefix so it generates user queries; 4M raw -> 300K selected; models matched Llama-3-8B-Instruct with 10M-point official tuning. [S] https://arxiv.org/abs/2406.08464 . Fit: good for general-assistant replay data (see 2.4), not for domain-specific ops items.
- **Evol-Instruct** (WizardLM): iterative rewriting to harder/diverse instructions. [S] https://arxiv.org/pdf/2304.12244 (2023-04, older; its gains were measured with GPT-4-judged evals).
- **Rejection sampling / RFT**: sample N responses, keep those that pass a checker/reward. For us the checker can be programmatic (promtool for PromQL syntax, `--help` text for flag existence, a live Prometheus for query results). [S for the technique: https://aiunderstanding.org/learn/rejection-sampling-fine-tuning (secondary); I for the checkers.] Strongest available lever: it makes targets verified rather than merely plausible. [I]
- Self-distillation (sampling targets from the model being tuned, filtered) keeps the output distribution on-policy and limits forgetting: Amazon Nova self-distilled reasoning recovered reasoning almost entirely (see 3.3). [S, summary] https://aihub.hkuspace.hku.hk/?p=7754 (secondary).

### 2.4 Forgetting: replay and control examples

- Our own lesson (targeted data without controls took alert rules 78% -> 11%) is the textbook case; the cure is mixing in examples of every behavior to preserve (replay). [I from our evidence; general support: Biderman "forgets less" is about LoRA's regularization, not a replacement for replay.]
- Context-free synthetic data (model's own generations on generic prompts) mixed into fine-tuning preserved reasoning in thinking models. [S] https://arxiv.org/html/2505.13811v1 (summary).
- Recommendation form: for each weakness-targeted slice, add >= 1:1 controls from the *other* 6 sets' task types, and keep a fixed per-skill quota. [I]

### 2.5 Template exactness, loss masking, packing

- **Chat template exactness**: the template at training must match vLLM's serving template byte-for-byte, including the empty think block placement and `enable_thinking` handling. Qwen3's own training kept an empty `<think></think>` in non-thinking samples so format is consistent, and used `/think` `/no_think` flags. [S] https://arxiv.org/pdf/2505.09388 (summary via search; I did not read the full report). [I] A mismatch in whitespace or generation prompt is a silent distribution shift; diff rendered strings from training vs vLLM `apply_chat_template` on 20 items.
- **Completion-only / assistant-only loss**: TRL `assistant_only_loss=True` (needs `{% generation %}` markers in the chat template) and `completion_only_loss` for prompt-completion data. [S] https://huggingface.co/docs/trl/main/en/sft_trainer (summary). Tool-call turns, tool results (loss should be OFF on tool outputs) and the empty think block need an explicit decision. [I]
- **Packing**: `position_ids` alone do not prevent cross-example attention; padding-free packing is correct only on FlashAttention varlen kernels with `cu_seqlens` giving block-diagonal masks. [S, summary of a forum/commit write-up] https://huggingface.co/datasets/John6666/forum2/commit/f6b864135197ae52b141b3baad5f2d83d2bc7a37 (weak source; verify in TRL docs).
- **For Mamba/DeltaNet layers, packing needs the recurrent state reset at each example boundary**, not just an attention mask: PackMamba and the mamba issue tracker show it is possible via `cu_seqlens`/position indices but "somewhat tricky". [S] https://arxiv.org/pdf/2408.03865 ; https://github.com/state-spaces/mamba/issues/180 . Whether our trainer does this for Lightning and for Qwen3.8 DeltaNet is [U]. If packing is on and not state-resetting, the model trains on leaked context across examples. Cheapest safe default: no packing for the hybrid models, or verify with a unit test that packed vs unpacked loss match. [I]

---

## 3. Hybrid think/no-think models

- **How Qwen3 did it**: Stage-3 "thinking mode fusion" = continual SFT on the reasoning-RL model with a mixed dataset of thinking data (rejection-sampled from the model itself) and non-thinking data with an **empty think block retained**; `/think` and `/no_think` flags randomly inserted, response follows the last flag; some thinking samples have no flag since thinking is the default. [S] https://arxiv.org/pdf/2505.09388 (summary).
- **Our empty-think-only data is a one-sided version of that.** [I] The adapter learns "always empty think", which is consistent with measured weakness on rows that need reasoning (our observation).
- **Reasoning-trace collapse** (2026): standard SFT on answer-only data rapidly suppresses valid reasoning traces across four open reasoning models while final answers stay plausible; answer-only metrics hide it; simple loss-masking strategies mitigate it without teacher traces. [S] https://arxiv.org/abs/2605.21127 (abstract read). One secondary summary reports valid-reasoning rate falling to exactly zero for a Qwen3-4B-Thinking fine-tune on answer-only data. [S, secondary] https://www.crusoe.ai/resources/blog/preserving-the-trace-a-guide-to-fine-tuning-chain-of-thought-models . Another study: vanilla SFT dropped Math from 70% to 6% (different setup; do not transfer the number). [S, summary] https://arxiv.org/html/2411.15382v2 / MetaGDPO.
- **Vendor guidance**: Unsloth says fine-tuning a hybrid on a non-reasoning dataset "may affect its reasoning ability"; to keep it use a mix, ~75% reasoning / 25% non-reasoning; if reasoning is not needed, omit it. [S] https://docs.unsloth.ai/basics/qwen3-how-to-run-and-finetune and https://unsloth.ai/docs/models/qwen3.5/fine-tune . The 75/25 ratio is vendor heuristic, not a measured optimum. [I]
- **Options, in order of cost**: (a) Serve with thinking off and accept the loss (current). (b) Add a minority of think-on examples whose traces are the model's own (self-distillation / rejection-sampled and verified), tagged by the template flag so the mode is switchable. (c) Distill traces from a larger teacher; Nemotron post-training sets provide reasoning on/off pairs (see 4). Licensing note: that dataset's synthetic content was produced by Qwen and DeepSeek models and the card requires honoring those licenses if the trained model is distributed. [S] https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2 (read).
- **Eval must report think-on and think-off separately** and include a structural metric (non-empty valid reasoning rate), not only answer accuracy. [S for the metric idea: 2605.21127; I for application]
- Nemotron 3.5 Lightning: reasoning toggled with `enable_thinking` in the chat template; NVIDIA documents LoRA/SFT via NeMo Automodel and Megatron Bridge and RL via NeMo RL/NeMo Gym. [S, secondary] https://www.beri.net/learning/nemotron-3-5-lightning-model-card (summary). Whether those toolchains handle our 24 GB budget is [U].

---

## 4. Tool-use and agentic fine-tuning

### 4.1 Practices

- **When (not) to call**: NVIDIA's When2Call (NAACL 2025) tests four behaviors: call tool, ask follow-up, say unable, hallucinate-answer. Finding: SOTA tool-callers had much room to improve, and **preference optimization with its MCQ-derived pairs improved more than traditional SFT**. [S] https://arxiv.org/abs/2504.18851 (summary). This maps directly to our "decline / ask / say I don't know" trap items. [I]
- **Multi-turn trajectories with tool results and errors**: Nemotron-Agentic data are synthetic multi-turn user/agent/tool trajectories scored and filtered by LMs. [S, secondary] https://hyper.ai/en/datasets/nemotron-agentic-tool-use-v1 . Training on recovery requires examples where the tool returns an error or empty output and the correct next turn is retry/ask/stop; I did not find a canonical "recovery data" recipe paper ([U]); the Terminal-Pivot set below includes "recovering from injected faults". [S]
- **Mask tool outputs from loss** (they are environment text) and keep the tool-call turn in loss. [I]
- **Verification before training**: APIGen-style three-stage checking (format, execution, semantic) is the published template for verifiable function-call data. [S, summary] https://github.com/apigen-mt/apigen-mt.github.io (secondary landing; APIGen paper not opened).
- **Do not use BFCL as training data**: it is an eval (Apache 2.0 data/code; V4 adds web search, memory and format sensitivity). Using it to train removes it as a trustworthy external check. [S for contents/license: https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html , summary; I for the advice.]

### 4.2 Openly licensed datasets (listed only, nothing downloaded)

Licenses/sizes from search or page reads on 2026-10-04; verify the card before any use.

| Dataset | License | Size | Fit for our task |
|---|---|---|---|
| nvidia/When2Call (SFT + preference + test) [S, read] https://huggingface.co/datasets/nvidia/When2Call | CC-BY-4.0 | 15,000 SFT; 9,000 preference; 3,652 MCQ test; 300 LLM-judge test | **High** for decline/ask/unknown behavior and a DPO/KTO-style pass. Domain is generic APIs, not ops; need translation to bash/PromQL contexts. Keep the test split out as an external eval. |
| nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1 [S, read] https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1 | CC-BY-4.0 | 31,111 samples; 630 unique terminal tasks | **High** for terminal/ops and RL with verifiers (`terminus_judge`); prompts average ~39,900 chars, far too long for 24 GB training at full length. Use as a source of task ideas or for short-subset RFT. |
| nvidia/Nemotron-Agentic-Tool-Use-v1 [S, secondary] https://hyper.ai/en/datasets/nemotron-agentic-tool-use-v1 | CC-BY-4.0 | 335,122 (19,028 interactive agent; 316,094 tool calling) | Medium-high: multi-turn tool calling at scale; sample a few thousand for replay. Verify license on the HF card ([S] only via a mirror site). |
| nvidia/Nemotron-Post-Training-Dataset-v2 [S, read] https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2 | Mostly CC-BY-4.0, small ODC-BY / CC-BY-SA parts; gated; Qwen/DeepSeek-license obligations on distribution | 1M-10M (math 239K, code 175K, STEM 355K, chat 628K, multilingual) | Medium: reasoning on/off pairs for replay (section 3). Not tool-specific. |
| Team-ACE/ToolACE [S, summary] https://huggingface.co/datasets/Team-ACE/ToolACE | Apache-2.0 (per search summary; verify) | ~10K dialogs, 26,507 APIs | Medium: diverse function calling incl. non-call turns; generic APIs. |
| NousResearch/hermes-function-calling-v1 [S, summary] https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1 | Apache-2.0 (per search summary; verify) | ~12K | Medium-low: JSON-schema calls, includes error-style files (a `func-calling-error` file was listed). Hermes tag format differs from each model's native template; re-render. |
| Salesforce/xlam-function-calling-60k (APIGen) [S, summary] https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k | **CC-BY-NC-4.0**, gated | 60,000 | Low-medium: non-commercial; single-turn verified calls. Fine for private lab research, not for anything we might redistribute. |
| BFCL (gorilla-llm/Berkeley-Function-Calling-Leaderboard) | Apache-2.0 | n/a | **Eval only**; keep out of training. |

Gap: I found no openly licensed PromQL-instruction/alert-rule dataset ([U], not exhaustively searched). Prometheus docs/tests and our own validated pairs remain the source.

---

## 5. Beyond SFT on one 24 GB GPU

### 5.1 Preference methods

- DPO is the default for modest paired data; KTO needs only binary good/bad labels; SimPO drops the reference model (about half the memory of DPO) and is length-normalized; ORPO merges SFT and preference loss. [S, summary; weak] https://jacar.es/en/alignment-evaluation-rlhf-dpo-and-recent-alternatives/ . I did not find a primary head-to-head under our conditions ([U]).
- **Why it fits us**: our hardest category is calibration (decline vs call vs ask). That is a choice between behaviors, and When2Call's authors found preference optimization beat SFT for exactly this. [S] arXiv 2504.18851. With a LoRA adapter, TRL can use the adapter-disabled base as the reference (no second copy), so memory is near SFT. [I; TRL behavior not verified in this research]
- Pairs are cheap for us: chosen = verified-correct behavior, rejected = the confabulating/over-calling output our own adapters produced in eval. [I]
- Caveat: DPO can increase verbosity and drift off-format; always include format controls and re-run the full 7-set suite. [I]

### 5.2 GRPO / RLVR

- TRL GRPOTrainer: loss types `dapo` (default), `dr_grpo`, `sapo`, `cispo`; vLLM colocate mode by default; LoRA/PEFT supported for memory; async custom reward functions; built-in tool/environment support (`tools=`, `environment_factory`); default `num_generations=8`; importance-sampling correction for vLLM-train mismatch; `beta=0.0` (no reference model) by default. Supported list includes Qwen 2.5/3/3.5/3.6. [S] https://huggingface.co/docs/trl/main/en/grpo_trainer (read). Whether Qwen3.8 and Nemotron hybrid Mamba MoE work in TRL/Unsloth GRPO is [U].
- Unsloth claims GRPO in one 24 GB GPU and large VRAM reductions. [S, vendor claim] https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl .
- **Evidence RLVR works for execution-checkable structured output**: text-to-SQL with GRPO and execution reward (Arctic-Text2SQL-R1 at 7B; Think2SQL; FINER-SQL at 0.5-3B). [S, summary] https://arxiv.org/pdf/2504.15077 , search results above.
- **What RLVR does and does not do**: pass@1 improves but pass@k at large k does not exceed the base; RLVR mostly sharpens what the base already samples. [S] https://neurips.cc/virtual/2025/poster/119944 (Yue et al., 2025). Implication: RLVR will not inject facts either, but it can raise reliability on tasks where the base sometimes gets it right. [I]
- **Fit to our tasks [I, recommendation not measured]**:
  - PromQL: reward = `promtool` parse (cheap, local) AND result match against a fixture Prometheus with known series; flag lookup: reward = flag in the tool's captured `--help` text; traps: reward = correct behavior class, scored by a rule (regex on tool-call vs question vs "don't know").
  - Cost on 24 GB: generation dominates. Qwen3.8-27B is too big for colocated vLLM + trainer on 24 GB; Lightning (3B active, ~30B total) in 4-bit is plausible only with sleep mode; both unverified ([U]). A tractable first RL run is a small dense model or the 8B-class models we already use. [I]
  - Reward hacking risks: syntactically valid but trivially constant PromQL; flags that exist but are wrong for the intent; "decline always" collapses. Mitigate with mixed reward (validity + semantic result + non-trivial), balanced traps, KL or small LR. [I]
  - Sequence: SFT/RFT first, DPO on calibration pairs second, RLVR only if the remaining failures are checkable and the base already hits them at k>1. [I]

### 5.3 2026 status

TRL has first-class GRPO with tool/environment hooks and several loss variants; NVIDIA ships NeMo RL/Gym with the Nemotron 3.5 release; Unsloth markets 24 GB GRPO. [S] sources above. Practical maturity for hybrid Mamba/DeltaNet MoE on a laptop 5090 under QLoRA is [U].

---

## 6. Eval methodology

### 6.1 What to adopt (and what we already do)

- **Paired differences, clustered SEs, power analysis, variance reduction** are Miller's recommendations ("Adding Error Bars to Evals", 2024-11). Cluster adjustment can inflate SEs by up to 3x when items are correlated. [S] https://arxiv.org/abs/2411.00640 and https://anthropic.com/research/statistical-approach-to-model-evals (summary). Our items inside a set are likely clustered by template/generator, so item-level independence is optimistic. [I]
- **Do not use the CLT intervals at small n or near 0/1 accuracy**: they under-cover and collapse at perfect scores; use Bayesian or exact/bootstrap intervals (Wilson/Clopper-Pearson for single proportions). [S] https://arxiv.org/abs/2503.01747 (Bowyer et al., ICML 2025).
- **Sign test on paired items is appropriate**; Holm is valid under any dependence (FWER); BH is for FDR and is more powerful if we are willing to tolerate some false discoveries. [S for the generic statistics: standard; I for our choice]. Define the family once before looking at results (7 sets x arms), or Holm is meaningless. [I]

### 6.2 Repeats and temperature-0 nondeterminism

- Temperature 0 is not deterministic on GPU servers: batch-dependent kernels make outputs depend on server load; three batch-invariant kernels gave bitwise reproducibility across 1,000 runs. [S, summary] Thinking Machines "Defeating Nondeterminism in LLM Inference" https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ (found via secondary coverage; primary page not opened).
- vLLM `VLLM_BATCH_INVARIANT=1`: requires compute capability >= 8.0 (3090 is 8.6, 5090 laptop is 12.0; the doc says >= 8.0), explicitly validated on DeepSeek, Qwen3, Qwen2.5, Llama 3, GPT-OSS, Mistral, Phi; performance cost; beta; LoRA, Mamba and MoE not addressed. [S] https://docs.vllm.ai/en/v0.26.0/features/batch_invariance/ (read). Whether 0.29.0 supports our hybrid models and adapters is [U]; test by running the same eval twice at different concurrency and diffing outputs.
- Consequence for repeats: with temperature 0 and a fixed seed, "3 repeats" may mostly measure server-load noise, not sampling variance. They are still valuable as a noise floor, but they are **not independent samples of items**. Aggregate repeats to one score per item (mean or majority) BEFORE the sign test; running the test on 3x rows inflates n. [I]
- Also measure the **seed/training-run variance**: a single training run per arm cannot show that a win is not run-to-run noise (our "single-run wins did not replicate" lesson). Retrain with >= 3 seeds on the key comparison. [I] The vLLM-nondeterminism blog implies eval noise, not training noise.

### 6.3 Minimum detectable effect vs item count [A]

Exact two-sided binomial sign test on discordant pairs (items where the arms differ), arithmetic computed by hand:

- If a set has 20 discordant pairs (a ~70-item set, the average; ours range from 9 to 158): 15-5 gives p about 0.041 (unadjusted), 16-4 about 0.012, 17-3 about 0.0026.
- Holm across 7 sets means the smallest p must beat 0.05/7 = 0.0071: 17-3 passes, 16-4 does not.
- So per ~70-item set, only about an 85% vs 15% split among discordant items is detectable under Holm; real effects of ~5-10 points of accuracy will usually be invisible per set. Pool across sets for one pre-registered primary metric (item-level, clustered by set) and treat the per-set results as exploratory. [I based on A]
- Our small sets are worse than the average: to clear the strictest Holm step (0.05/7) any set needs at least 9 discordant items all going one way. alert (n=9) can only pass at 9-0 (8-1 gives p 0.039); trap3 (n=12) needs 11-1; promqlcat (n=18) needs 16-2 even if every item is discordant. A 30-item alert set would pass at 24-6 (p 0.0014). [A]
- Items, not repeats, drive power. Adding repeats does not add items; adding new fresh items does. [I]

### 6.4 Contamination, held-out, splits

- Maintain three disjoint pools: train, dev (for LR/checkpoint/early-stop choices), and a locked test set evaluated once per decision milestone. Every checkpoint selection on the 478-item suite makes it a dev set. [I; general ML practice, standard]
- Generate fresh items per milestone from a *different* generator/prompt than training data (see 2.2). [I]
- Add an external eval you did not write (BFCL subsets, When2Call test split) to catch in-house overfitting. [S for availability: sections 4.1 and 4.2; I for the use]

### 6.5 LLM-as-judge pitfalls

- Documented biases: position, verbosity, self-preference (judges favor own-family outputs, including GPT-4o and Claude 3.5 Sonnet in a 2025 study). [S, summary] https://arxiv.org/pdf/2411.15594 , https://arxiv.org/pdf/2410.02736 , https://arxiv.org/html/2406.07791v5 .
- For our task most checks are programmatic (flag exists in `--help`; promtool parse; trap = behavior class). Prefer programmatic scoring; use a judge only for open-ended rows, with swapped order, a judge from a different family than any model that generated training data, and human audit of a sample. [I]
- Scorer drift: our own hand audits found scorer errors; re-audit scorers per release and log scorer version in results. [I from the brief]

---

## 7. Antipractices and outdated approaches

| # | Antipractice | Why | Source |
|---|---|---|---|
| 1 | Using SFT to inject facts | Slower to learn, then raises hallucination; models learn facts in pretraining | Gekhman et al. https://aclanthology.org/2024.emnlp-main.444 [S] |
| 2 | Attention-only LoRA | Underperforms MLP/MoE-inclusive LoRA at equal parameter count | https://thinkingmachines.ai/blog/lora/ [S] |
| 3 | Same LR as full FT (or a copied 2e-4 without a sweep) | LoRA needs ~10x; LR is the most sensitive knob | Thinking Machines; https://arxiv.org/abs/2601.22708 [S] |
| 4 | Trusting low training loss | Loss in bits/token near 0 = memorization of low-entropy synthetic targets; it is not an eval | inference from Thinking Machines capacity framing [I]; our own 0.0003 case |
| 5 | Answer-only (empty-think) SFT on a reasoning model with answer-only metrics | Reasoning-trace collapse is hidden by accuracy | https://arxiv.org/abs/2605.21127 [S] |
| 6 | Weakness-targeted data without controls | Forgetting of untargeted skills; our alert-rule collapse 78% -> 11% | our measurement; general: replay [I] |
| 7 | n-gram-only decontamination | Rephrased/translated leakage passes | https://arxiv.org/pdf/2311.04850 [S] |
| 8 | Packing with only position_ids, or packing hybrids without state reset | No cross-example isolation in attention; SSM/DeltaNet state leaks | https://arxiv.org/pdf/2408.03865 ; mamba issue 180 [S]; TRL discussion (weak) |
| 9 | Treating repeats of a temperature-0 run as independent samples | Items are the unit; server-load noise is not sampling noise | Miller https://arxiv.org/abs/2411.00640 [S]; I |
| 10 | CLT error bars on small sets or near 0/100% | Under-covers, interval collapses at the extremes | https://arxiv.org/abs/2503.01747 [S] |
| 11 | Single-run "wins" | Run-to-run variance; need seeds and paired tests | our measurement; Miller [S] |
| 12 | Tuning on the eval suite | Eval becomes dev set; wins are selection effects | standard practice [I] |
| 13 | QLoRA on hybrid/MoE without checking vs bf16 | Vendor reports higher quantization difference for Qwen3.5-class | https://unsloth.ai/docs/models/qwen3.5/fine-tune [S, vendor] |
| 14 | NEFTune, rsLoRA, DoRA, PiSSA as default | Gains reported mostly on chat win-rates or at high rank; tuned vanilla LoRA matches most variants | https://arxiv.org/abs/2601.22708 [S]; NEFTune [U on replication] |
| 15 | GPT-4-judge-only evals (AlpacaEval-style) for structured tasks | Length/self-preference bias; unnecessary when checks are programmatic | https://arxiv.org/pdf/2410.02736 [S] |
| 16 | Training on the benchmark you report (BFCL etc.) | Destroys external validity | [I] |
| 17 | Treating RLVR as capability expansion | Pass@k at large k is bounded by base | https://neurips.cc/virtual/2025/poster/119944 [S] |
| 18 | LIMA-style "1,000 is enough" for tool precision | LIMA is about style alignment, 2023, evaluated by humans on chat; says nothing about exact flag/query correctness | LIMA NeurIPS 2023 [S]; I |
| 19 | Large effective batch with LoRA | LoRA loses more at big batch | Thinking Machines [S] |
| 20 | Epochs > 3 on small synthetic sets | Overfitting; templates memorized | Unsloth guide [S, vendor] |

Items marked "possibly outdated": LIMA (2023), Evol-Instruct (2023), rephrased-samples paper (2023), rsLoRA (2023). Mechanisms still cited in 2025-26 sources but not re-validated here.

---

## 8. Ranked top 10 changes for our setup

These are recommendations, not measured results. "Cost" = cost on one 24 GB GPU unless otherwise stated.

1. **Add a pre-registered primary metric and pool across sets; aggregate repeats to per-item scores; report exact/bootstrap CIs.** Effect: stops over-reading per-set noise (per ~70-item set only about 17-3 splits clear Holm, see 6.3). Cost: zero GPU, an afternoon of scripting. Evidence: Miller https://arxiv.org/abs/2411.00640 ; Bowyer https://arxiv.org/abs/2503.01747 ; arithmetic [A].
2. **Lock a fresh held-out test set from a generator different from training data and compute max train-similarity (embedding) per eval item.** Effect: removes the largest hidden risk (in-house leakage, selection on the 478). Cost: embeddings on CPU/small GPU, plus fresh item authoring. Evidence: https://arxiv.org/pdf/2311.04850 ; inference on shared generators.
3. **Replace empty-think-only data with a mixed-mode set** of think-on examples carrying the model's own verified traces plus think-off examples, template flags honored; evaluate think-on and think-off separately plus a valid-reasoning-rate metric. The ratio is open, and the vendor points the other way from a minority: Unsloth says to keep a minimum of 75% reasoning examples to preserve reasoning (https://unsloth.ai/docs/models/qwen3.5/fine-tune, verified 2026-10-04). Our G6q set has 0% (audit fix 4). Either follow the vendor (>=75% think-on) or pre-register a ratio sweep (e.g. 25 / 50 / 75%) with the valid-reasoning-rate metric as a gate; do not pick 20-25% on intuition. Expected effect: recover reasoning-dependent rows without hurting think-off ones. Cost: generation time on the existing model (serve on the 3090) plus ~25-30% more training tokens. Evidence: Qwen3 report https://arxiv.org/pdf/2505.09388 ; https://arxiv.org/abs/2605.21127 ; Unsloth guidance (vendor heuristic).
4. **Mandatory controls in every targeted dataset (fixed per-skill quotas, replay of the other six sets' task types) plus a regression gate on all 7 sets before accepting any adapter.** Expected: prevents repeats of the 78% -> 11% collapse. Cost: more training tokens only. Evidence: our measurement; replay literature summarized in 2.4.
5. **LR sweep (3 values, 1 seed) before any variant, then confirm winner with 3 seeds; keep r=16, alpha=32, all-linear including MoE experts and Mamba projections, effective batch <=32, <=3 epochs.** Expected: the largest single quality lever per Thinking Machines/2601.22708; seeds measure run variance. Cost: each Lightning run is small (3B active); 27B runs cost the most, so sweep on Lightning first. Evidence: https://thinkingmachines.ai/blog/lora/ ; https://arxiv.org/abs/2601.22708 .
6. **Add rejection-sampled, programmatically verified targets (promtool parse + fixture-Prometheus result match; flag in `--help`) and dedup (exact + MinHash + embedding clustering with per-template caps).** Expected: higher precision on PromQL/flag rows, less template collapse. Cost: CPU plus inference time on the 3090; no extra training cost. Evidence: SemDeDup/MinHash recipe https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/index.html ; RFT general technique (secondary).
7. **Import When2Call (15K SFT + 9K preference, CC-BY-4.0), re-rendered into bash/PromQL contexts, for decline/ask/unknown behavior; run a small DPO/KTO pass on pairs built from our own confabulating outputs.** Expected: the best-evidenced lever for trap rows; SFT alone underperformed preference tuning in the paper. Cost: DPO with an adapter on one GPU is near SFT memory [I, unverified in TRL]; a few hours. Evidence: https://arxiv.org/abs/2504.18851 .
8. **Verify packing and loss masking on both hybrids**: unit test that packed loss equals unpacked loss; confirm state reset at example boundaries; mask tool outputs; diff the training-rendered chat string against vLLM's `apply_chat_template` output for 20 items. Expected: removes silent training bugs; may change results either way. Cost: tiny. Evidence: https://arxiv.org/pdf/2408.03865 ; TRL SFT docs (summary).
9. **Check the Qwen3.8 NF4 adapter against a higher-precision base at serving time (e.g. FP8/bf16 base on the 3090 or CPU offload), and test `VLLM_BATCH_INVARIANT=1` on both models with the adapters.** Expected: tells us whether quantization mismatch or serving nondeterminism is hiding gains. Cost: serving only; the 27B bf16 base does not fit one 24 GB card, so FP8 or the two-GPU pool is needed. Evidence: Unsloth QLoRA warning (vendor, Qwen3.5) https://unsloth.ai/docs/models/qwen3.5/fine-tune ; vLLM doc https://docs.vllm.ai/en/v0.26.0/features/batch_invariance/ . Applicability to Qwen3.8 is [I].
10. **Pilot RLVR (TRL GRPO, LoRA, `loss_type=dr_grpo`, verifier rewards) on a narrow, checkable slice (PromQL validity+result, flag existence) with a small model first.** Expected: improves pass@1 on slices the base already sometimes solves; will not add facts. Cost: highest of the ten (generation-heavy; 27B colocated vLLM+trainer does not fit 24 GB; hybrid-model support unverified [U]). Evidence: https://huggingface.co/docs/trl/main/en/grpo_trainer ; https://neurips.cc/virtual/2025/poster/119944 ; text-to-SQL results https://arxiv.org/pdf/2504.15077 . Do this only after items 1-7 stabilize the baseline.

### Not recommended now (rationale in section 7)
NEFTune, DoRA/PiSSA/OLoRA swaps, rank above 16, attention-only targeting, expanding the dataset with xLAM (non-commercial) before dedup/control fixes, LLM-judge scoring for programmatically checkable rows.

### Open unknowns
Expert-layer LoRA coverage in our Lightning pipeline; Qwen3.8 NF4 vs bf16 gap; vLLM 0.29.0 batch-invariance on hybrid models with adapters; GRPO support for hybrid Mamba/DeltaNet MoE; whether an open PromQL instruction dataset exists; NEFTune replications.

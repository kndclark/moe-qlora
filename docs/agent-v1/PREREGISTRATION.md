# agent-v1: pre-registration of opening 1

Committed before the opening starts. Nothing below may change once the first
run has been made; anything learned afterwards goes in the results write-up,
marked as such.

## The set

30 real lab tasks, sealed on the desktop at `ssh llm:~/sealed/agent-v1`.
They were written on 2026-10-09 by an author agent that never opened
moe-qlora `results/` or any dev item.

- Kinds: 8 flag-lookup, 8 promql-alert, 7 triage, 7 config-edit.
- Needs: 15 shell (read-only), 7 shell plus file edits, 5 PromQL, 3 PromQL
  plus file edits.
- Every task can run on either node. Per-task timeouts run 300–600 s and sum
  to 15,600 s.
- Each task is `prompt.md`, a `fixture/` copied in as the working directory,
  and `check.sh`. The check prints PASS (exit 0) or one `FAIL: reason` line
  (exit 1). Live-value tasks use tolerances.
- `MANIFEST.sha256` (189 files, copied here) has sha256 `e5644438…6b703`
  (`e56444386126026662161c3e64de6722d6d5512d6e21bd56a3064bb1604b6703`).
  The set's `TASKS.sha256` has sha256
  `a95e0dfe1aa170ef083cc1b388b8fdbc92bb63f85af1a62283be43f89e9f7044`.
  `sha256sum -c` passed on all 189 files on 2026-10-09, before this opening.
- An opening is any run that shows a model the prompts, or anyone reading
  prompts, answers, references or FAIL reasons. The set retires after its
  third opening. Every opening gets a row in `OPENINGS.md` (copied here and
  kept in the set).

## The questions

1. **Harness.** Should the lab's agent be Codex CLI, or a minimal loop of our
   own?
2. **Model.** Should `agent` keep serving G6u, or Lightning without the
   adapter?

## Arms

There are 8 arms: harness {codex, loop} × model {g6u, base} × thinking
{off, on}. Each runs every task once (240 runs, one sample each).

- **Models.** g6u is `lightning-g6u` (Nemotron 3.5 Lightning NVFP4 plus the G6u
  LoRA); base is `lightning`. Both come from one vLLM session on the desktop
  (gpu-lab `lab up agent`, vllm-openai v0.29.0, max-model-len 131072,
  `qwen3_coder` tool parser, `nemotron_v3` reasoning parser). Both harnesses
  call it directly on :8300, not through LiteLLM. Neither harness sets
  sampling, so both get the server's defaults (temperature 1.0, top_p 0.95,
  from Lightning's generation_config).
- **Thinking off is primary**, as the hand-off pre-registered: G6u's best
  measured mode is thinking off. Thinking on is secondary. Every off run is
  made before any on run.
- **Order.** Tasks are shuffled with seed 20261009, the same order in both
  modes. Within a task, the four arms of a mode start at a position that
  rotates from task to task.
- **codex.** Codex CLI rust-v0.162.1:
  `codex exec --json --ephemeral -s workspace-write`. CODEX_HOME is one
  fixed path, emptied and rewritten before every run, so nothing on PATH
  (Codex puts its `apply_patch` helper there) names a task or an arm.
  - It speaks the Responses API and uses its own prompt and its own tools.
  - Reasoning effort is `none` for off and `medium` for on.
  - Context window 131072, auto-compact at 117,964 tokens, web search off,
    sandbox network on.
  - Plugins off. With them on, each fresh CODEX_HOME clones the plugin
    marketplace from the network in the background.
  - Codex has no metadata for the lab's model names. It warns ("Model
    metadata for `lightning-g6u` not found") and uses its fallback.
- **loop.** The loop from `bench/research_eval.py` `run_item`, over
  `/v1/chat/completions`.
  - Its one tool is the `bash` tool from `training/tools.py`, and prompt.md is
    the only message. Thinking is set with `enable_thinking`.
  - vLLM parses the tool calls: `run_item`'s own parser misses Lightning's XML
    calls.
  - Command output is capped at 20,000 characters (head and tail), with a
    120 s limit per command and 60 calls per run.
- **Deadline.** For both harnesses, a run's deadline is the task's
  `timeout_s`.
- **Runner.** gpu-lab `bench/agent_ab.py`, with `bench/agent_ab_bwrap.sh`,
  at commit b1258d8.

## The shell both harnesses get

Every command from either harness runs in Codex's Linux sandbox (bubblewrap,
workspace-write, network on).

- **Environment.** Neither shell sees any variable whose name contains
  KEY, SECRET or TOKEN. Codex's `shell_environment_policy` excludes them,
  and the loop's runner drops them. Without the policy, Codex's shell saw
  the lab key in a dummy run.
- **Writes.** The root is read-only. A command can write only to the
  workspace and to a private `/tmp`, which is emptied between runs.
  - The workspace is a fresh copy of the task's `fixture/`, at the same path
    (`/tmp/agentab/work`) in every run. Nothing in the model's context names
    the task or the arm. One exception: the codex arm's config file is
    readable from its shell and names the served model.
- **The wrapper.** `agent_ab_bwrap.sh` stands in for bwrap.
  - It binds the GPU device nodes, so `nvidia-smi` works.
  - It keeps the host pid namespace and shows the host `/proc` read-only, so
    `ps` sees the machine.
  - It hides `~/sealed`, the results directory, `~/.ssh` and its own
    directory behind empty tmpfs mounts.
  - It reads the paths to hide from a file beside itself, never from the
    environment. Codex hands bwrap only the environment its policy leaves.
    In a dummy run on 2026-10-13, that policy stripped the wrapper's path
    variables, and Codex's shell listed `~/sealed` and wrote to the host's
    `/tmp`. With no list, or nowhere to mount it, the wrapper now refuses to
    run.
  - Codex picks the first bwrap on PATH that the sandbox cannot write. One
    under `/tmp` is passed over for the system bwrap, which hides nothing,
    so the wrapper lives in `~/.cache`. Codex runs the link's target and
    checks that `--help` lists `--as-pid-1`, so the wrapper is a copy, not a
    link, and passes `--help` through.
  - Before the first run, the runner checks from inside the sandbox that
    every hidden path is empty and `/tmp` is private. Otherwise it refuses
    to start.
- **Caveat: docker.** The docker socket stays reachable, because a triage
  task may need `docker logs`, so an agent could change containers. The runner
  checks the model server's `/health` before every run and waits for it to
  return. A run during which the server went down counts as infra, not as a
  FAIL.
- **Caveat: shared read-only state.** The read-only root is the live
  machine. Anything another process changes between runs (metrics, logs)
  differs between arms of the same task. The rotation spreads that drift
  across arms.

## Metric and grading

- **Metric.** The pass rate under `check.sh`.
- **Grading.** Each run is graded as soon as it ends, with cwd at the set
  root and `PROM_URL=http://lab-desktop:9090`. The grader gets the final
  message as the answer file and the workspace after the run.
- **Selftest.** Each task's reference answer (`selftest/`) is checked before
  the opening and again after it.

## Voids and reruns

- **Voids.** A task is void for every arm if either:
  - its reference answer fails the selftest before or after the opening; or
  - review of its FAILs shows the check or the live data broke independently
    of the answers (for example, a query that returns nothing for the
    reference too).
- **Infra.** A run where the server was down, or refused or dropped the
  request, is rerun once. A second infra makes that task void for the pair
  of arms being compared.
- **Review.** Every FAIL gets one class from its run record: wrong answer,
  no tool use, answer format, timeout, context exhausted, harness error, or
  check/data broken.

## Decision rules

The test is exact two-sided McNemar on discordant pairs, α = 0.05, over the
tasks that are not void. With about 30 tasks only lopsided splits reach
significance (arithmetic): 6–0 gives p = 0.031, 8–1 gives p = 0.039, and 7–1
gives p = 0.070.

1. **Harness.** Codex at its configured default (thinking on, G6u) is compared
   with the better loop arm for G6u: off or on, whichever passes more, with a
   tie going to off. Codex becomes the lab's harness unless that loop arm
   passes significantly more.
2. **Model.** In the harness that rule 1 picks, at its mode, G6u stays
   `agent` unless base passes significantly more.
3. **Everything else is descriptive.** That covers the thinking-off
   contrasts, which are primary for G6u's training regimen but decide nothing
   here.

## Prior evidence, not from the set

Smoke tests on 2026-10-09 put Codex with Lightning, thinking off, at 0/6 tool
calls end to end (G6u and base), and 0/3 when the captured request was
replayed. With thinking on it was 3/3. Replays traced the cause to Codex's
~17k-character developer prompt: with it dropped, 2/3 called tools. So the
expectation for codex-*-off is near zero on any task that needs the shell.
That expectation is why rule 1 uses Codex's thinking-on default.

## What gets published after

Per-arm counts, the McNemar lines, voids and FAIL classes, and the ledger
row. Prompts, answers, references and FAIL reasons are not published.

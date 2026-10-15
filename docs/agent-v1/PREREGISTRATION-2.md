# agent-v1: pre-registration of opening 2

Final, 2026-10-14. The two open decisions were set by David before the first
run and are recorded below: 5 replicates per task per arm, and base becomes
`agent` unless G6u passes significantly more. Nothing in this file may change
from here; anything learned afterwards goes in the results write-up, marked
as such.

The set, its seal, its 30 tasks and the rule that counts openings are
unchanged: see `PREREGISTRATION.md`. This is the second of the set's three
openings. The 2026-10-14 review of two keys (below) belongs to opening 1 and
costs nothing here, because it opened only tasks that opening 1 had already
opened.

## What opening 1 settled

`RESULTS-1.md` has the write-up.

- **Harness, decided.** Codex CLI is the lab's harness. Against the better
  loop arm for G6u it led 7 discordant tasks to 4 (p = 0.549), which is not a
  loss, and rule 1 asked only that the loop not beat it.
- **Model, undecided.** In Codex with thinking on, the configured default,
  base passed 25 of 30 tasks and G6u 20 of 30: 6 tasks base-only, 1
  G6u-only, exact two-sided McNemar p = 0.125 over 7 discordant pairs. Rule 2
  needed significance, so G6u stayed `agent`.
- **The pattern rule 2 could not act on.** Base led all four of its
  contrasts with G6u, and the thinking-off pair significantly. One sample per
  task per arm cannot separate that from run-to-run noise, which is what this
  opening is for.

## The question

One question. Under Codex with thinking on, in the shell described below,
should `agent` serve base (`lightning`) or keep serving G6u
(`lightning-g6u`)?

Rule 1 is settled, so the loop arms are gone. Thinking off is not re-run:
opening 1 measured it on both models and it is not what the lab serves.

## Arms

Two arms: `codex-base-on` and `codex-g6u-on`.

**Replicates per task per arm: 5** (300 runs). Opening 1's two arms each took
about 11 minutes for their 30 runs (median run 9 s and 12 s, slowest 210 s),
so 5 replicates is about 2 hours, with a worst case of 43 hours if every run
hits its timeout. The cheaper option was 3 (180 runs, about 70 minutes),
which fixes the single-sample problem too; 5 was chosen because a majority of
5 absorbs one noisy replicate per arm, and because an odd count cannot tie.

Power does not come from more tasks: the set is 30 tasks and all 30 are
already used. It comes from repeating each task, so that a per-task verdict
is a majority rather than a coin flip.

- **Models.** base is `lightning` (Nemotron 3.5 Lightning NVFP4); g6u is the
  same weights plus the G6u LoRA. Both come from one vLLM session on the
  desktop (gpu-lab `lab up agent`, vllm-openai v0.29.0, max-model-len 131072,
  `qwen3_coder` tool parser, `nemotron_v3` reasoning parser), called directly
  on :8300, not through LiteLLM. Sampling is the server's default
  (temperature 1.0, top_p 0.95), unchanged, so replicates differ.
- **codex.** As in opening 1: Codex CLI rust-v0.162.1,
  `codex exec --json --ephemeral -s workspace-write`, reasoning effort
  `medium`, context window 131072, auto-compact at 117,964 tokens, web search
  off, sandbox network on, plugins off. CODEX_HOME is one fixed path, emptied
  and rewritten before every run; it now sits under the sandbox state
  directory rather than the lab user's cache.
- **Order.** Tasks are shuffled with seed 20261014. The two arms of a task
  run back to back, and which one leads alternates from task to task, so
  drift in the live machine falls inside a pair rather than between arms. The
  whole 30-task pass then repeats, re-shuffled from the same generator, until
  the replicate count is met.
- **Deadline.** A run's deadline is the task's `timeout_s`, as in opening 1.
- **Runner.** gpu-lab `bench/agent_ab.py` with `bench/agent_ab_bwrap.sh` at
  commit `68722e9`, "Run opening 2's arms with replicates", pushed on branch
  `agent-user` and checked out at that sha on the serving node for the run.
  The ledger row records that sha; if the branch later lands on `main`, the
  row records the merge commit beside it.
- **Commands.** The opening is these four:

      agent_ab.py selftest --set DIR --out DIR --tag before
      agent_ab.py run      --set DIR --out DIR --reps 5 --seed 20261014 \
                           --arms codex-g6u-on,codex-base-on
      agent_ab.py selftest --set DIR --out DIR --tag after
      agent_ab.py summary  --out DIR --reps 5 --arms codex-g6u-on,codex-base-on

  `run --dry-run` prints the order and stops, opening nothing. Given opening
  1's seed, modes and `--reps 1` it reproduces that opening's recorded
  240-run order exactly, and `summary` over opening 1's rows reprints that
  opening's published lines unchanged, so the replicate machinery added for
  this opening leaves a single pass as it was.

## The shell both arms get

Every command runs in Codex's Linux sandbox (bubblewrap, workspace-write,
network on), as in opening 1, with one change: the shell no longer runs as
the lab user.

Precondition: `lab install` has run on the serving node, so the sandbox user
and its state directory exist, and `agent_ab.py preflight` passes. **A failed
preflight aborts before any run, and since no model has seen a prompt, that
is not an opening.**

- **Its own uid.** Commands run as `labagent`, a system user with no home, no
  login shell, and no group but its own. Opening 1 showed why: a shell
  carrying the lab user's uid reaches whatever that uid may reach. One run
  removed the serving container through the container socket, which that uid
  may use by group membership, and another signalled a lab service by pid.
  Both are uid checks in the kernel, so a uid of its own closes them.
- **What that closes, by permission, not by configuration.** The container
  socket (0660 root:docker, and this uid is in no such group), signals to the
  lab user's processes, that user's 0700 directories, and the environments of
  its running processes, so the lab key is out of reach even if the
  environment policy missed it.
- **What it keeps.** The host pid namespace, the host `/proc` read-only, and
  the GPU device nodes, so `ps` and `nvidia-smi` still see the machine. The
  set's tasks assume a read-only shell on the real machine, and that is
  unchanged.
- **HOME** is the private `/tmp`, emptied between runs, because the lab
  user's home is not readable by this uid. No task's prompt refers to a home
  directory.
- **Traversal** of the lab user's home is granted to that one uid by ACL, not
  by loosening the directory's mode, so no other account on the machine gains
  anything. Files the tasks read under that home, such as the gpu-lab
  checkout, stay readable.
- **Environment** crosses the uid change explicitly: the wrapper clears it
  and re-sets each variable it was handed except any whose name contains KEY,
  SECRET or TOKEN. Codex passes bwrap no environment of its own and the uid
  switch drops it, so without this the shell would start bare.
- **Timeouts** sweep by uid. A process group owned by another uid cannot be
  killed by the runner's group kill, and Codex puts its child in a new
  session anyway, so the runner kills by uid after each run and after each
  timeout. Verified: a 300 s command under a 15 s limit returns at 15 s and
  leaves nothing behind.
- **The workspace** is still a fresh copy of the task's `fixture/` at the
  same path in every run, now shared with the sandbox uid by ACL rather than
  handed over, because Codex removes and recreates paths inside it.
- **preflight** asserts each of these from inside the sandbox before the
  opening starts: the uid, the container socket shut, the lab user's
  processes not signallable, no KEY/SECRET/TOKEN in the environment, nothing
  readable from the runner's own process environment, every hidden path
  empty, `/tmp` private, and the machine still visible (at least 20 host pids
  and at least 1 GPU).
- **Caveat: the container socket is now shut.** Opening 1 left it reachable
  in case a triage task needed container logs. No task needed them: no prompt
  in the set asks for container logs, no run of the 240 called for them, and
  the three checks that use containers run as the grader, outside the
  sandbox. No passing answer came from the socket; what did come of it was
  the removed server above. The aggregate is in the sealed review. If a
  future task needs logs, the way to give them is a filtering read-only
  socket proxy, not group membership.
- **Caveat: shared read-only state.** The read-only root is the live machine,
  so anything another process changes between runs differs between runs and
  between arms. Pair-adjacent order and the alternating leader spread that;
  replication measures what is left.
- **Caveat: the config file.** The codex arm's config file is readable from
  its own shell and names the served model. This is the one thing in the
  shell that differs between the two arms by design, as in opening 1, and it
  cannot be hidden without breaking Codex's re-exec of its own sandbox.

## Metric and grading

- **Metric.** The pass rate under `check.sh`, as in opening 1. Each run is
  graded as it ends, with cwd at the set root and
  `PROM_URL=http://lab-desktop:9090`.
- **Per-task verdict.** A task passes for an arm if a majority of that arm's
  replicates pass.
- **Amendment 1, t28.** One recorded amendment applies from this opening on.
  It lives in the sealed set, not in this repo, because its text contains
  answer tokens; it is pre-registered here by hash only. sha256
  `d22d464f546f91adbd2541d787527e664000af5527077082c79e2c49a5da9300`. It
  widens one triage task's accepted vocabulary. It is applied mechanically to
  the recorded answers after grading: the regrades land in the run
  directory's `amend.jsonl`, one `{task, arm, rep, pass, why}` line each, and
  `summary` applies them and reports how many changed a verdict. It edits no
  set file, and changes no
  opening 1 verdict (re-graded, opening 1's rule 1 reads 7–3, p = 0.344, and
  rule 2 is unmoved). `MANIFEST.sha256` and `TASKS.sha256` verified unchanged
  after it was written, on 2026-10-14: 189 files, 0 failures.
- **Key review, carried out under opening 1.** t16's key was reviewed and
  cleared: it accepted the one correct answer and rejected the other seven
  for defects the sealed review records. No change to that task.
- **Selftest.** Each task's reference answer is checked before the opening
  and again after it.

## Voids and reruns

- **Voids.** A task is void for both arms if its reference answer fails the
  selftest before or after the opening, or if review of its FAILs shows the
  check or the live data broke independently of the answers.
- **Infra.** A run where the server was down, or refused or dropped the
  request, is rerun once. If either arm of a task ends with fewer valid
  replicates than the pre-registered count, that task is void for the
  comparison.
- **Review.** Every FAIL gets one class from its run record: wrong answer, no
  tool use, answer format, timeout, context exhausted, harness error, or
  check/data broken. Reviewing this opening's FAILs is part of this opening.

## Decision rules

The test is exact two-sided McNemar on the per-task majority verdicts,
α = 0.05, over the tasks that are not void. With about 30 tasks only lopsided
splits reach significance (arithmetic): 6–0 gives p = 0.031, 8–1 gives
p = 0.039, 7–1 gives p = 0.070. Opening 1's codex-on pairs were 1–6, so if
G6u's single discordant win was noise, 0–6 is significant and 1–7 is not.

**Rule 2, the burden of proof: base becomes `agent` unless G6u passes
significantly more.** Chosen because base led every contrast in opening 1 and
won the one that reached significance, and because serving base takes the
adapter out of the serving path. The alternative was the conservative shape
of opening 1's rule 2, G6u stays unless base wins significantly; with 30
tasks that would most likely return "not significant" a second time, and with
one opening left the set would retire with the question open. The rule is
symmetric in procedure; only its direction differs, and the direction is
fixed here, before the first run.

Concretely, on the per-task majority verdicts over the tasks that are not
void: if G6u passes significantly more than base, `agent` keeps serving G6u.
In every other outcome, including "not significant", `agent` serves base and
the G6u adapter leaves the serving path.

Everything else is descriptive and changes nothing: per-arm pass rates with
intervals, how many tasks flipped verdict between replicates (the noise this
opening exists to measure), FAIL classes, the amendment's effect, and
McNemar on the first replicate alone, which is the comparison opening 1 made.

One secondary test is pre-specified so that it cannot be chosen afterwards: a
sign-flip permutation test on the per-task difference in pass counts, 100,000
flips, seed 20261014. If the primary test is not significant and this one is,
it is recorded as suggestive and the served model does not change.

## Contamination

This file is never in a model's context: the runner reads it, the model does
not. Two sessions have read items, both recorded against opening 1 in
`OPENINGS.md`: opening 1's runner, which reviewed the FAILs of six tasks, and
the 2026-10-14 key review of two tasks. Neither wrote a task and neither
wrote training data, which is the rule the ledger exists to check — the set
is worth running only while no model has been trained toward it.

## What gets published after

Per-arm counts, the McNemar line, the permutation line, the replicate flip
counts, void and FAIL-class counts, and the ledger row. Prompts, answers,
references and FAIL reasons are not published.

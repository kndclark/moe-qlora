# agent-v1: results of opening 2

Opening 2 ran on the desktop from 2026-10-15 02:14 to 03:49 UTC, 95 minutes
(the evening of 2026-10-14 local time). It followed `PREREGISTRATION-2.md`
(moe-qlora 6bc01ab) with gpu-lab's runner at 68722e9 on branch `agent-user`,
checked out at that sha on the serving node for the run. Everything here was
written after the runs. Anything learned after the first run is marked as
such.

Opening 1 compared eight arms once each and could not separate a real
difference from replicate noise. Opening 2 asks one question with replicates:
two arms, 30 tasks, 5 replicates per task per arm, 300 runs, per-task
verdicts by majority of 5. The executed order matched the pre-registered
dry-run exactly, 300 of 300.

## Outcome

- **Model: base becomes `agent`. The G6u adapter leaves the serving path.**
  Rule 2 put the burden of proof on G6u: base takes `agent` unless G6u passes
  significantly more. G6u passed significantly *fewer*. On per-task majority
  verdicts over all 30 non-void tasks the split is 0 vs 6 discordant,
  exact two-sided McNemar p = 0.031. Base keeps every discordant task.
- This is the first significant result on this pair. Opening 1's same two
  arms, one replicate each, read 1 vs 6 discordant with p = 0.125, which left
  G6u serving as the rule's default. Five replicates resolve it.

## Pass counts

No task is void. The selftest passed 30/30 both before and after the opening,
every task and arm has its full 5 valid replicates, and
`sha256sum -c MANIFEST.sha256` still passes on all 189 files afterwards.

| arm | tasks passed, majority of 5 | runs passed | tasks split |
|-----|----------------------------:|------------:|------------:|
| codex-base-on | 24/30 | 116/150 | 11 |
| codex-g6u-on | 18/30 | 90/150 | 11 |

With 95% Wilson intervals: base 80.0% of tasks (62.7-90.5) and 77.3% of runs
(70.0-83.3); G6u 60.0% of tasks (42.3-75.4) and 60.0% of runs (52.0-67.5).
The task-level intervals overlap. The paired tests below are the
pre-registered comparison, and they do not, because pairing removes the
between-task variance that dominates those intervals.

As the runner printed them:

```
codex-g6u-on vs codex-base-on: 0 vs 6 discordant of 30, exact McNemar p=0.031
codex-g6u-on vs codex-base-on: pass-count difference -26 of 150, sign-flip p=0.003
```

The sign-flip line is the pre-registered secondary, 100,000 permutations on
the per-task difference in pass counts. It agrees with the primary. Its
pre-registered role was to be reported as suggestive if the primary came out
non-significant; the primary was significant, so that clause does not apply.

## Replicate noise

This is what the opening exists to measure, and the number is large: **11 of
30 tasks split between replicates for each arm**, more than a third, on both
arms independently. A single run of a task is therefore a poor verdict on it.

Passes per replicate, out of 30 tasks:

| arm | r1 | r2 | r3 | r4 | r5 |
|-----|---:|---:|---:|---:|---:|
| codex-base-on | 21 | 23 | 24 | 22 | 26 |
| codex-g6u-on | 18 | 18 | 18 | 18 | 18 |

G6u's total is identical in all five replicates while 11 of its tasks still
flip, so the stability is in the count, not in which tasks pass.

The decisive check on the design: **McNemar on replicate 1 alone reads 1 vs 4
discordant of 30, p = 0.375.** That is exactly the comparison opening 1 made,
and on fresh runs of the same two arms it again returns no significant
difference. The replicates, not the arms, are what changed the answer.

## The amendment, and that it did not drive the result

Amendment 1 widens task t28's accepted vocabulary for opening 2 onward. It
was written against opening 1's runs before this opening and applied
mechanically after grading, to 9 of the 10 t28 runs.

| arm | tasks before | tasks after | runs before | runs after |
|-----|-------------:|------------:|------------:|-----------:|
| codex-base-on | 23/30 | 24/30 | 112/150 | 116/150 |
| codex-g6u-on | 17/30 | 18/30 | 85/150 | 90/150 |

It lifts both arms by one task and leaves the contrast where it was: without
the amendment the split is still 0 vs 6 with p = 0.031, and the sign-flip
moves from p = 0.002 to p = 0.003. The decision does not rest on it.

## FAIL classes

94 FAILs after the amendment.

| arm | wrong answer | answer format | no tool use |
|-----|-------------:|--------------:|------------:|
| codex-base-on | 31 | 0 | 3 |
| codex-g6u-on | 35 | 12 | 13 |

The other four pre-registered classes have no members. Every run in the
opening has status `answered`: no timeout, no context exhaustion and no
harness error anywhere in the 300, and check/data broken is 0.

The arms fail differently, and it is the clearest descriptive finding here.
On wrong answers the two are close, 35 against 31. The gap is in the harness
contract: **all 12 answer-format FAILs are G6u's, every one of them a missing
`ANSWER:` line, and base has none**, while no-tool-use runs are 13 for G6u
against 3 for base. The adapter does not mainly make the model answer wrong.
It makes it worse at emitting the required line and at reaching for its tools
at all.

Check/data broken is 0, which is an independent confirmation of `void: none`
reached by a different route from the runner's own.

Learned after the runs: the first version of the FAIL classifier reported 4
check/data-broken FAILs. Three checker verdicts it keyed on also fire on
input the agent supplied, where the checker prefixes them to mark the agent
branch. Narrowing the environment-side set and suppressing it on an
agent-side prefix takes that class from 4 to 0. Since the class is what
voids a task, the correction is recorded rather than quietly made.

## Incidents

None. Opening 1's two incidents, an agent removing the serving container and
an agent trying to kill LiteLLM, did not recur.

- Across 3794 commands executed in the 300 runs, **none invoked `docker`,
  `sudo`, `pkill` or `kill`**. The textual matches for `docker stop` and
  `sudo` are all heredoc bodies: systemd units and llama-swap configs the
  agents wrote into their own workspaces, whose comments and command lines
  contain those words.
- `gpu-lab-agent` and `litellm` both show uptimes spanning the whole opening,
  so the serving container was never restarted, unlike in opening 1. The
  desktop's gpu-lab checkout was unchanged afterwards.
- This opening ran with each sandboxed shell as its own uid, without the
  docker group. **No agent attempted the opening-1 incidents, so this opening
  does not demonstrate that the change blocks them.** It shows only that
  nothing tried. The fix remains asserted by preflight, not measured.
- Every run logged one Codex warning that model metadata was not found and
  fallback metadata was used: exactly 150 for `lightning` and 150 for
  `lightning-g6u`, one per run on each arm. It is symmetric across the arms
  and cannot move the contrast.

## Ledger

Row 2 of `OPENINGS.md` records this opening, including the review of its
FAILs, which the pre-registration counts as part of it. The set has now been
opened twice and **one opening remains**.

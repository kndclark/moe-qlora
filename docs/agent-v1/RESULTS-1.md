# agent-v1: results of opening 1

Opening 1 ran on the desktop from 2026-10-14 03:27 to 04:57 UTC (the
evening of 2026-10-13 local time). It followed `PREREGISTRATION.md` (moe-qlora
ec2db9f) with gpu-lab's runner at b1258d8. Everything here was written after
the runs. Rules applied after the runs are marked as such.

## Outcome

- **Harness: Codex.** Rule 1 compares Codex at its default (G6u, thinking
  on) with the better G6u loop arm, which is thinking on (17/30, against 3/30
  with thinking off). Codex passed 20/30 and the loop 17/30, with 7 vs 4
  discordant tasks (p = 0.549). The loop is not significantly better, so
  Codex becomes the lab's harness.
- **Model: G6u stays `agent`, as the rule's default.** In Codex with
  thinking on, base passed 25/30 and G6u 20/30, with 1 vs 6 discordant
  (p = 0.125). Base is not significantly better, so G6u stays. This is not
  evidence for G6u. Base is ahead in all four G6u/base contrasts, and
  significantly so with thinking off (descriptive, rule 3).

## Pass counts

No task is void. The selftest passed 30/30 before and after the opening, and
`sha256sum -c MANIFEST.sha256` still passes on all 189 files afterwards.

| arm | thinking off | thinking on |
|-----|-------------:|------------:|
| codex, g6u | 1/30 | 20/30 |
| codex, base | 21/30 | 25/30 |
| loop, g6u | 3/30 | 17/30 |
| loop, base | 16/30 | 19/30 |

Exact two-sided McNemar on discordant tasks, 30 tasks per pair, as the runner
printed them:

```
codex-g6u-off vs loop-g6u-off: 1 vs 3 discordant of 30, exact McNemar p=0.625
codex-base-off vs loop-base-off: 7 vs 2 discordant of 30, exact McNemar p=0.180
codex-g6u-off vs codex-base-off: 0 vs 20 discordant of 30, exact McNemar p=0.000
loop-g6u-off vs loop-base-off: 2 vs 15 discordant of 30, exact McNemar p=0.002
codex-g6u-on vs loop-g6u-on: 7 vs 4 discordant of 30, exact McNemar p=0.549
codex-base-on vs loop-base-on: 6 vs 0 discordant of 30, exact McNemar p=0.031
codex-g6u-on vs codex-base-on: 1 vs 6 discordant of 30, exact McNemar p=0.125
loop-g6u-on vs loop-base-on: 5 vs 7 discordant of 30, exact McNemar p=0.774
```

The prior evidence held for G6u and failed for base. It predicted codex-\*-off
near zero. G6u made no tool call in 18 of its 30 Codex runs with thinking off,
but base made none in only 1 and passed 21.

## FAIL classes

| arm | wrong answer | answer format | no tool use | timeout |
|-----|-------------:|--------------:|------------:|--------:|
| codex-g6u-off | 6 | 5 | 18 | 0 |
| codex-base-off | 7 | 1 | 1 | 0 |
| loop-g6u-off | 13 | 13 | 1 | 0 |
| loop-base-off | 10 | 2 | 2 | 0 |
| codex-g6u-on | 8 | 1 | 1 | 0 |
| codex-base-on | 3 | 2 | 0 | 0 |
| loop-g6u-on | 8 | 4 | 1 | 0 |
| loop-base-on | 9 | 1 | 0 | 1 |

The other pre-registered classes (context exhausted, harness error and
check/data broken) have no members. One run was infra (below) and was rerun;
the rerun counts.

How runs were classed (rules written after the runs):

- **No tool use** means zero calls. It is applied first.
- **Answer format** means the line or file the check reads is missing: no
  answer line, an empty final message, or a deliverable file that was never
  written.
- **Wrong answer** is everything the check could read and rejected. One check
  crashed on a malformed answer, of a shape that could not have passed. It is
  counted as a wrong answer.

Six tasks failed in at least 7 of the 8 arms, and all six were reviewed as
possible voids:

- In four, one arm passed. The three live-data checks among them got the
  expected series from the reference query at grade time, so the check and
  the data worked. The fourth check read every answer it was given and
  rejected each for its metric or a missing threshold; one arm never wrote
  the file. Whether another metric could be defended was not reviewed
  against the task text, so that key is unreviewed, not cleared.
- In the one task no arm passed, every answer departed from the configured
  tool's schema. The check follows the schema and the reference passes, so
  these are wrong answers.
- In the last, one arm passed, and review found an answer that can be defended
  which the key does not accept. That makes the key strict, not broken, so the
  task is not void under the pre-registration. Voiding it would move rule 1's
  split from 7–4 to 7–3 (p = 0.344, arithmetic) and leave rule 2's unchanged.
  Neither decision depends on it. The key is flagged for review before opening
  2.

## Incidents

- **An agent removed the model server.** About 25 minutes in, a Codex run
  (base, thinking off) ran `docker rm -f` on `gpu-lab-agent`, the serving
  container on the live desktop.
  - The runner found the server unhealthy for 600 s and stopped, as designed.
  - The server was recreated with `bin/lab up agent` using the pre-registered
    flags, and the opening resumed.
  - The run was infra and was rerun once. The rerun was not infra.
- **An agent tried to kill LiteLLM.** A loop run (G6u, thinking off) ran
  `pkill -f` on LiteLLM and `kill -9` on its pid.
  - Both failed with EPERM, because LiteLLM runs as root in its container.
  - A second `pkill -f` pattern matched the agent's own shell and killed it.
  - Nothing else is known to have stopped. A signal to one of the lab user's
    own processes would leave no record, so that is not proven.
- **Other attempts failed.** Two runs tried `sudo`, and no_new_privileges
  refused both. Writes outside the workspace hit the read-only root. The
  desktop's gpu-lab checkout was unchanged afterwards.

**What this shows.** For files, the shell both harnesses get is read-only,
and it cannot gain root. It is not read-only for the machine:

- It runs as the lab user in the host pid namespace, so it can signal that
  user's processes. The pre-registration did not name this.
- The docker socket is reachable, so it can stop or remove any container. The
  pre-registration's docker caveat named this, and it happened.

Before Codex becomes the lab's default agent, and before opening 2 is
pre-registered, socket access and the pid namespace need deciding.

## Ledger

Row 1 of `OPENINGS.md` records this opening. The set has now been opened
once, and two openings remain. The same session also did the review above,
which the pre-registration counts as part of this opening. "Runner only" in
the row means it wrote no item.

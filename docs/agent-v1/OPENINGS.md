# Openings ledger (retire after 3)

| # | date | who | purpose | arms |
|---|------|-----|---------|------|
| 1 | 2026-10-13 | Claude Code session cd10081e, as runner only | Pre-registered A/B (moe-qlora ec2db9f): harness and model for `agent` | 8: codex/loop × g6u/base × off/on |
| 2 | 2026-10-14 | Claude Code session c5429c94, as runner only | Pre-registered A/B with replicates (moe-qlora 6bc01ab, runner gpu-lab 68722e9): model for `agent` | 2: codex × g6u/base × on, 5 replicates |

Row 1's flagged key review was completed on 2026-10-14 by Claude Code
session ed3fc252, as runner only: it opened t16 and t28, wrote no task and
no training data, and recorded what it found in the set. That is part of
opening 1 under the rule above, because it opened no task opening 1 had not
already opened.

Row 2 ran the pre-registered two-arm comparison with five replicates per
task per arm, 300 runs, and its session also reviewed and classified every
FAIL it produced, which the pre-registration counts as part of the opening.
It wrote no task and no training data. Results are in `RESULTS-2.md`: base
becomes `agent` and the G6u adapter leaves the serving path.

The set has now been opened twice. One opening remains.

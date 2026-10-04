"""Thinking off against thinking on for one arm, through decision_rule.py unchanged.

noise_summary.path is remapped so that "base" reads ARM's thinking-on files and "off" reads
ARM's thinking-off files; the rule's "against base" CI is then off minus on, its row flags
are the rows thinking off loses, and its cost lines compare the two modes. Both modes need
every repeat of every set, as in decision_rule.py. Ignore its "qualifying" and "serve" lines.

usage: [REPS="r1 r2 r3"] python3 probes/mode_compare.py ARM    (ARM: base, g6q, g6u, ...)
"""
import os
import runpy
import sys

here = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(here)
if len(sys.argv) != 2:
    sys.exit(__doc__.split("usage: ")[1])
ARM = sys.argv[1]
os.environ["MODE"] = "nothink"  # decision_rule.py prints this mode; path() below picks per arm
sys.path.insert(0, here)
import noise_summary  # noqa: E402

lab = "" if ARM == "base" else f"-{ARM}"


def path(tag, label, rep):
    suf = "think-4k" if label == "base" else "nothink"
    if rep == "r1":
        return os.path.join(REPO, "results", f"research-eval-{tag}-lightning{lab}-{suf}.json")
    return os.path.join(REPO, "results", "noise", f"research-eval-{tag}-lightning{lab}-{suf}-{rep}.json")


noise_summary.path = path
print(f"{ARM}: 'base' = thinking on (think-4k), 'off' = thinking off")
sys.argv = ["decision_rule.py", "off"]
runpy.run_path(os.path.join(here, "decision_rule.py"), run_name="__main__")

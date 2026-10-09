"""Print doc rows by letter, each cut to N chars, or the part after a marker."""
import sys
doc = "/home/david/moe-qlora/.claude/worktrees/next-round/docs/laptop-memory-levers.md"
letters, n = sys.argv[1].split(","), int(sys.argv[2])
mark = sys.argv[3] if len(sys.argv) > 3 else None
for i, l in enumerate(open(doc), 1):
    for L in letters:
        if l.startswith(f"| {L} |"):
            s = l[l.find(mark):] if mark and mark in l else l
            print(f"--- {L} (line {i}, {len(l)} chars)\n{s[:n]}\n")

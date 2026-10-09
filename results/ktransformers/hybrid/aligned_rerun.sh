#!/bin/bash
# Re-run the THP and streams probes with 4 KiB-aligned experts; push fixed probes to the desktop
H=/home/david/handoff-0948db4b-files/hybrid
cd "$H" || exit 1
grep -n "^EXPERT =\|^CH =" h2d_*.py
for s in h2d_pool h2d_streams; do
  mv -f "laptop-$s.jsonl" "laptop-$s.misaligned.jsonl" 2>/dev/null
  bash sweep.sh "$H" laptop "$s"
done
timeout 30 scp -q h2d_load.py h2d_sweep.py h2d_streams.py h2d_pool.py sweep.sh hybrid_run.sh llm:kt-relu2-results/hybrid/ && echo desktop-copied
for s in h2d_pool h2d_streams; do echo "== $s"; cat "laptop-$s.jsonl"; tail -2 "laptop-$s.err"; done

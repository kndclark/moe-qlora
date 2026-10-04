#!/bin/bash
# L70 (docs/next-model-plan.md): untrained Llama-3.1-70B, served across both cards by gpu-lab
# `bin/lab pool up`, on the seven sets by S1's protocol (s1_screen.sh's ev() and flags), one
# pass: thinking off at 512 tokens, as Llama 3.1 has no thinking mode. The pool must already
# be up with the 70B; this script neither starts nor stops it.
# Outputs: results/research-eval-<set>-s1-<TAG>-nothink.json (+ .log). Existing evals are
#   skipped, so a rerun resumes.
# usage: [TAG=l70] [B=http://lab-desktop:8200] [LIMIT=n OUT=dir] [ARGS="..."] l70_eval.sh
#   LIMIT=n with OUT=dir is a smoke run: n items a split, written under OUT, not results/.
#   ARGS go to every g7a_eval.py call. L70m: TAG=l70m ARGS="--render llama31-meta-json".
set -u
TAG=${TAG:-l70} B=${B:-http://lab-desktop:8200}
MODEL=hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4
here=$(cd "$(dirname "$0")/.." && pwd)
# The harness runs help commands in its working directory, and `git diff -h` prints 34 lines
# inside a git repo but 130 outside, which adds 8 seen_tool items to v1 (166, not 158) and
# changes what a model's git lookups return. Every earlier eval ran inside a repo.
cd "$here" || exit 1
out=${OUT:-$here/results}
mkdir -p "$out"
python3 "$here/probes/g7a_eval.py" --selfcheck | tail -1 | grep -qx "selfcheck PASS" || { echo "g7a_eval.py --selfcheck failed"; exit 4; }
served=$(curl -sf -m10 $B/v1/models) || { echo "pool not answering at $B"; exit 2; }
echo "$served" | grep -q "\"$MODEL\"" || { echo "pool is not serving $MODEL"; exit 2; }
common=(--max-calls 3 --temperature 0 --window 4000 --seed 20260923 --concurrency 16 ${LIMIT:+--limit $LIMIT} ${ARGS:-})
ev() {  # label set [extra...]
  local label=$1 set=$2; shift 2
  if [ -f "$out/research-eval-$label.json" ]; then echo "  $label: exists, skipped"; return; fi
  if [ "$set" = promql ] && ! curl -sf -m3 http://lab-desktop:9090/-/ready >/dev/null; then
    echo "  $label: SKIPPED, desktop Prometheus unreachable (link down?)"; return; fi
  local t=$(date +%s)
  python3 "$here/probes/g7a_eval.py" --base $B --model "$MODEL" --label "$label" \
    --thinking off --max-tokens 512 --set "$set" "${common[@]}" "$@" \
    --out "$out/research-eval-$label.json" > "$out/research-eval-$label.log" 2>&1
  echo "  $label: exit $?, $(( $(date +%s)-t ))s"
}
sets=(${SETS:-v1:v1 v2:v2 rocky:rocky promqlcat:promql general:general alert:alert trap3:trap3})
for s in "${sets[@]}"; do
  tag=${s%%:*} set=${s##*:} extra=()
  [ "$tag" = promqlcat ] && extra=(--promql-catalog)
  ev "$tag-s1-$TAG-nothink" "$set" "${extra[@]}"
done

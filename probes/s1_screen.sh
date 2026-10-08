#!/bin/bash
# S1 (docs/next-model-plan.md): one base model, no adapter, on the lab's seven eval sets by
# G6's protocol (g6_eval.sh): thinking on at 4096 tokens (primary), thinking off at 512,
# --max-calls 3 --temperature 0 --window 4000 --seed 20260923 --concurrency 16, harness
# probes/g7a_eval.py (its --selfcheck gates the run). Before the evals it records vLLM's KV
# capacity and gpu-lab bench.py decode at c=1 and c=16, as l_single.sh does.
# NODE=laptop serves on 127.0.0.1:8304 (sm_120; platform profile max-power for the run).
# NODE=desktop serves on <desktop>:8304 (sm_86; free the card first: lab down --gpu-only).
# Outputs: results/research-eval-<set>-s1-<TAG>-{think-4k,nothink}.json (+ .log), and
#   results/s1/<TAG>/{serve.log,kv.txt,bench-c{1,16}.{json,log}}. Existing evals are
#   skipped, so a rerun resumes.
# usage: NODE=laptop|desktop TAG=name MODEL=repo REV=sha [EXTRA="..."] [UTIL=0.85]
#        [G7A="--think-tag"] [LOW_EFFORT=1] s1_screen.sh
#   EXTRA: more vLLM flags, split on spaces (on the desktop each word is %q-quoted).
#   G7A: more g7a_eval.py options for every eval of this model.
#   LOW_EFFORT=1 adds a third pass, thinking on with reasoning_effort low, labelled
#   <TAG>-low (Qwen3.8, whose template defaults to xhigh).
#   ADAPTER=dir serves a LoRA adapter as <TAG>, the base as <TAG>-base, and evaluates the
#   adapter (as g6_eval.sh serves its adapters: --max-lora-rank 16). On the desktop the
#   adapter is first copied to /tmp/s1-adapter-<TAG> there and its checksum compared.
#   REP=N files a repeat run under <TAG>-rN (labels, results/s1/, container); the served
#   name stays <TAG>. s1_compare.py reads it as model <TAG>-rN (or <TAG>-rN-low).
#   PASSES="think-low ..." runs only the named passes (think, nothink, think-low).
set -u
NODE=${NODE:?laptop or desktop} TAG=${TAG:?names the run} MODEL=${MODEL:?hf repo} REV=${REV:?pinned sha}
here=$(cd "$(dirname "$0")/.." && pwd)
addr=$(ssh -G llm | awk '/^hostname /{print $2}')   # the desktop's end of the direct link
out=$here/results
ftag=$TAG${REP:+-r$REP}
mkdir -p "$out/s1/$ftag"
name=s1-$ftag
if [ "$NODE" = desktop ]; then
  B=http://$addr:8304 port=$addr:8304:8000 run=(ssh llm sudo docker)
else
  B=http://127.0.0.1:8304 port=127.0.0.1:8304:8000 run=(docker)
fi
python3 "$here/probes/g7a_eval.py" --selfcheck | tail -1 | grep -qx "selfcheck PASS" || { echo "g7a_eval.py --selfcheck failed"; exit 4; }
stop() {
  "${run[@]}" logs $name > "$out/s1/$ftag/serve.log" 2>&1 || true
  "${run[@]}" rm -f $name >/dev/null 2>&1
  if [ "$NODE" = laptop ]; then
    echo performance | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null
    echo "profile: $(cat /sys/firmware/acpi/platform_profile)"
  fi
}
trap stop EXIT
[ "$NODE" = laptop ] && echo max-power | sudo -n tee /sys/firmware/acpi/platform_profile >/dev/null
extra=(${EXTRA:-})
if [ "$NODE" = desktop ]; then q=(); for w in "${extra[@]}"; do q+=("$(printf %q "$w")"); done; extra=("${q[@]}"); fi
served=$TAG mount=()
if [ -n "${ADAPTER:-}" ]; then
  src=$(realpath "$ADAPTER")
  if [ "$NODE" = desktop ]; then  # the container runs there: copy the adapter over, check it
    src=/tmp/s1-adapter-$ftag
    ssh llm mkdir -p $src && scp -q "$ADAPTER/adapter_config.json" "$ADAPTER/adapter_model.safetensors" llm:$src/ || exit 1
    [ "$(ssh llm sha256sum $src/adapter_model.safetensors | cut -d' ' -f1)" = \
      "$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)" ] || { echo "adapter copy differs"; exit 1; }
  fi
  served=$TAG-base mount=(-v "$src":/adapter:ro)
  extra+=(--enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules "$TAG=/adapter")
fi
"${run[@]}" run -d --name $name --gpus all --ipc=host -p $port \
  -v /srv/model-cache:/hf:ro "${mount[@]}" -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai:v0.29.0 \
  --model "$MODEL" --revision "$REV" --served-model-name "$served" \
  --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization ${UTIL:-0.85} \
  "${extra[@]}" >/dev/null || exit 1
echo "$NODE $ftag $MODEL@${REV:0:8} extra: ${EXTRA:-none} g7a: ${G7A:-none}"
t0=$(date +%s)
until curl -sf $B/health >/dev/null; do
  if [ "$("${run[@]}" inspect -f '{{.State.Running}}' $name 2>/dev/null)" != true ]; then
    echo "container exited before ready ($(( $(date +%s)-t0 ))s)"
    "${run[@]}" logs $name 2>&1 | grep -E "Error|error" | tail -3 | cut -c1-240; exit 2; fi
  [ $(( $(date +%s)-t0 )) -gt 900 ] && { echo "not ready in 900s"; exit 3; }
  sleep 10
done
echo "ready in $(( $(date +%s)-t0 ))s"
"${run[@]}" logs $name 2>&1 | grep -E "Available KV cache memory|GPU KV cache size|Maximum concurrency|Model loading took" \
  | sed 's/^.*\] //' | cut -c1-160 | tee "$out/s1/$ftag/kv.txt"
for c in 1 16; do
  [ -f "$out/s1/$ftag/bench-c$c.json" ] && continue
  python3 /home/david/gpu-lab/bench/bench.py --base $B --model "$TAG" --concurrency $c --repeats 2 \
    --label "s1 $ftag c=$c" --json-out "$out/s1/$ftag/bench-c$c.json" > "$out/s1/$ftag/bench-c$c.log" 2>&1
  echo "bench c=$c exit $?: $(grep -E 'decode rate' "$out/s1/$ftag/bench-c$c.log" | tail -1 | tr -s ' ')"
done

common=(--max-calls 3 --temperature 0 --window 4000 --seed 20260923 --concurrency 16 ${G7A:-})
ev() {  # label thinking max_tokens set [extra...]
  local label=$1 think=$2 mt=$3 set=$4; shift 4
  if [ -f "$out/research-eval-$label.json" ]; then echo "  $label: exists, skipped"; return; fi
  if [ "$set" = promql ] && ! curl -sf -m3 http://$addr:9090/-/ready >/dev/null; then
    echo "  $label: SKIPPED, desktop Prometheus unreachable (link down?)"; return; fi
  local t=$(date +%s)
  python3 "$here/probes/g7a_eval.py" --base $B --model "$TAG" --label "$label" \
    --thinking "$think" --max-tokens "$mt" --set "$set" "${common[@]}" "$@" \
    --out "$out/research-eval-$label.json" > "$out/research-eval-$label.log" 2>&1
  echo "  $label: exit $?, $(( $(date +%s)-t ))s"
}
sets=(v1:v1 v2:v2 rocky:rocky promqlcat:promql general:general alert:alert trap3:trap3)
passes=(think nothink)
[ "${LOW_EFFORT:-}" = 1 ] && passes+=(think-low)
[ -n "${PASSES:-}" ] && passes=($PASSES)
for p in "${passes[@]}"; do
  echo "== $p"
  for s in "${sets[@]}"; do
    tag=${s%%:*} set=${s##*:} extra=()
    [ "$tag" = promqlcat ] && extra=(--promql-catalog)
    case $p in
      think)     ev "$tag-s1-$ftag-think-4k" on 4096 "$set" "${extra[@]}" ;;
      nothink)   ev "$tag-s1-$ftag-nothink" off 512 "$set" "${extra[@]}" ;;
      think-low) ev "$tag-s1-$ftag-low-think-4k" on 4096 "$set" "${extra[@]}" \
                   --chat-kwargs '{"reasoning_effort":"low"}' ;;
      *)         echo "unknown pass: $p"; exit 1 ;;
    esac
  done
done

"""G6t (plan.md "G6t"), step 1: reasoning traces from base Lightning on G6q's own training
prompts, kept only where the eval's own scorer says the answer is right.

Why: G6q passes the gate but writes empty reasoning on 915 of 919 thinking-on turns;
base Lightning reasons on every turn (min 20 chars) and wins rocky_task (0.65 vs 0.45).
Lever (b): train the adapter on base's own reasoning, filtered for correctness.

How, reusing the eval harness so a trace is exactly what the eval would have seen:
  - each record's user prompt goes through bench/research_eval.py's run_item, with
    probes/g7a_eval.py's two patches (XML tool calls, open <think>), thinking on, the
    same tools (TOOLS, plus the record's extra_tools: promql), max 3 calls, window 4000;
  - tool calls run through research_eval's own execute (real `--help`/man pages, live
    Prometheus, web_search unavailable); a wrapper records each output, which run_item
    does not return, so the trace's history is rebuilt exactly as run_item sent it;
  - the answer is scored by research_eval's own score() on an eval-shaped item built from
    the record: cli_grounded -> held_out (the record's flag), compose -> two_flag (both
    flags), task_procedure -> task (every flag), trap_refusal/asserted_trap -> trap
    (the fake flag), alert_direct -> alert (G6q's series and checks), arithmetic ->
    no_tool (the record's number). promql_live has no single truth query, so it is checked
    here: promql calls only, and the answer carries every number of G6q's reference answer
    recomputed from Prometheus now (within 5% or 1), the same yes/no, or NODATA wording.

Accepted only if: status answered; every turn finished by `stop` and closed `</think>`
with non-empty reasoning; at most 3 calls; no claimed lookup that did not run; the
record's tool looked up (flag and trap kinds); every turn's prompt + completion fits
MAX_LEN - 1 tokens; and the kind's success test above. Only "default" (thinking-on)
records of those eight types are tried; everything else keeps its G6q record.

usage: python3 probes/g6t_collect.py --base URL --out FILE [--limit N] [--seed S]
       [--attempts K]   attempt 0 is greedy, as the eval; later ones sample (0.6, 0.95).
Resumes: records already in --out are skipped.
"""
import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
import types

sys.dont_write_bytecode = True
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import g7a_eval  # noqa: E402  (imports research_eval as rev)
import g6q_build  # noqa: E402

rev = g7a_eval.rev
rev.parse_tool_call = g7a_eval.parse_tool_call
rev.split_think = g7a_eval.split_think
g7a_eval.THINK_OPEN[0] = True

REPO = os.path.dirname(here)
DATASET = os.path.join(REPO, "results", "research_dataset_g6q.json")
PROM = "http://lab-desktop:9090"
MAX_LEN = 2048
ELIGIBLE = {"cli_grounded": "held_out", "compose": "two_flag", "task_procedure": "task",
            "trap_refusal": "trap", "asserted_trap": "trap", "alert_direct": "alert",
            "arithmetic": "no_tool", "promql_live": "promql"}

_local = threading.local()
_execute = rev.execute


def recording_execute(name, args, bins, window, prom=None):
    out, rec = _execute(name, args, bins, window, prom)
    _local.outputs.append(out)
    return out, rec


rev.execute = recording_execute


def item_for(i, r):
    split = ELIGIBLE[r["type"]]
    q = r["messages"][0]["content"]
    it = {"id": f"g6q-{i}", "split": split, "question": q}
    if split in ("held_out", "two_flag", "task", "trap"):
        it["tool"] = r["tool"]
        it["help_cmd"] = r.get("command") or (r.get("commands") or [f"{r['tool']} --help"])[0]
    if split == "held_out":
        it["aliases"] = [r["flag"]]
    elif split == "two_flag":
        it["groups"] = [[r["flag"]], [r["flag2"]]]
    elif split == "task":
        it["groups"] = [[f] for f in r["flags"]]
    elif split == "trap":
        it["fake_flag"] = r["fake_flag"]
    elif split == "alert":
        spec = g6q_build.ASPECS[r["spec"]]
        it.update(alertname=spec["name"], series=spec["series"], checks=spec["checks"])
    elif split == "no_tool":
        n = re.findall(r"-?\d[\d,]*(?:\.\d+)?", r["messages"][-1]["content"])[-1]
        it["expect"] = r"(?<![\d.])" + re.escape(n) + r"(?![\d])"
    return it


def numbers(text):
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)]


def promql_ok(r, run, final):
    if not run["calls"] or any(c["name"] != "promql" for c in run["calls"]):
        return False, "non-promql call"
    spec = g6q_build.PSPECS[r["spec"]]
    res = []
    for q in spec["queries"]:
        got, err = rev.prom_query(PROM, q)
        res.append([(m["metric"], m["value"][1]) for m in (got or [])])
    ref = spec["answer"](res)
    if not any(res):
        return bool(rev.NODATA.search(final)), f"nodata ref: {ref}"
    yn = re.match(r"\W*(yes|no)\b", ref, re.I)
    if yn:
        m = re.match(r"\W*(yes|no)\b", final, re.I)
        if not m or m.group(1).lower() != yn.group(1).lower():
            return False, f"yes/no differs from ref: {ref}"
    have = numbers(final)
    for x in numbers(ref):
        if not any(abs(y - x) <= max(1.0, 0.05 * abs(x)) for y in have):
            return False, f"missing {x} from ref: {ref}"
    return True, f"ref: {ref}"


def collect(a, i, r, docs, bins, attempt):
    it = item_for(i, r)
    ns = types.SimpleNamespace(
        model=a.model, base=a.base, tools=True, thinking="on", max_calls=3, max_tokens=a.max_tokens,
        temperature=0.0 if attempt == 0 else 0.6, top_p=None if attempt == 0 else 0.95,
        sample_seed=None if attempt == 0 else a.seed + attempt, window=4000, prom=PROM,
        set="promql" if r["type"] == "promql_live" else r["type"],
        promql_tool=(r.get("extra_tools") or [None])[0])
    _local.outputs = []
    run = rev.run_item(ns, it, bins)
    outputs = _local.outputs
    why = []
    turns = run["turns"]
    if run["status"] != "answered":
        why.append(f"status {run['status']}")
    if any("error" in t for t in turns):
        why.append("turn error")
    texts = [t.get("text", "") for t in turns]
    for t, tx in zip(turns, texts):
        if t.get("finish_reason") != "stop":
            why.append(f"finish {t.get('finish_reason')}")
        if "</think>" not in tx or not tx.split("</think>")[0].strip():
            why.append("no reasoning")
        if t.get("prompt_tokens", 0) + t.get("completion_tokens", 0) > MAX_LEN - 1:
            why.append(f"too long {t.get('prompt_tokens', 0) + t.get('completion_tokens', 0)}")
    final = run["final"] or ""
    if r["type"] == "promql_live":
        ok, note = promql_ok(r, run, final)
        s = {"promql_ok": ok, "note": note}
        if not ok:
            why.append(note)
    else:
        s = rev.score(it, run, docs)
        kind = rev.SPLIT_KIND[it["split"]]
        if s.get("claims_unrun_lookup"):
            why.append("claims unrun lookup")
        if kind in ("flag", "trap") and not s.get("looked_up"):
            why.append("tool not looked up")
        if kind == "flag" and not s.get("hit_and_grounded"):
            why.append("not hit_and_grounded")
        if kind == "trap" and not s.get("denied_heuristic"):
            why.append("trap not denied")
        if kind in ("alert", "no_tool") and not s.get("correct"):
            why.append("not correct")
        if kind == "no_tool" and run["calls"]:
            why.append("tool call on a no-tool record")
    # the history exactly as run_item sent it: visible prefix + call, then the output
    msgs = [{"role": "user", "content": it["question"]}]
    if len(outputs) != len(run["calls"]):
        why.append(f"outputs {len(outputs)} != calls {len(run['calls'])}")
    for k, c in enumerate(run["calls"]):
        visible, _ = rev.split_think(texts[k])
        parsed = rev.parse_tool_call(visible)
        if parsed is None or parsed[0] != c["name"] or parsed[1] != c["args"]:
            why.append(f"turn {k}: call does not re-parse")
            break
        cid = f"call_{k}"
        msgs.append({"role": "assistant", "content": parsed[2], "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "name": c["name"], "content": outputs[k] if k < len(outputs) else ""})
    return {"index": i, "type": r["type"], "attempt": attempt, "accepted": not why, "why": why,
            "status": run["status"], "calls": run["calls"], "score": s, "messages": msgs, "texts": texts,
            "tokens": [t.get("prompt_tokens", 0) + t.get("completion_tokens", 0) for t in turns],
            "prompt_tokens": [t.get("prompt_tokens") for t in turns],
            "completion_tokens": [t.get("completion_tokens") for t in turns],
            "reasoning_chars": [len(tx.split("</think>")[0].strip()) if "</think>" in tx else None for tx in texts]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8303")
    ap.add_argument("--model", default="lightning-nvfp4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="pilot: first N eligible after a seeded shuffle")
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    records = json.load(open(DATASET))
    elig = [i for i, r in enumerate(records) if r.get("thinking") == "default" and r.get("type") in ELIGIBLE]
    if a.limit:
        import random
        rng = random.Random(a.seed)
        by = {}
        for i in elig:
            by.setdefault(records[i]["type"], []).append(i)
        per = max(1, a.limit // len(by))  # the pilot samples every type
        elig = sorted(j for v in by.values() for j in rng.sample(v, min(per, len(v))))
    done = {}
    if os.path.exists(a.out):
        for line in open(a.out):
            x = json.loads(line)
            done.setdefault(x["index"], []).append(x)
    docs = rev.Docs()
    bins = {records[i]["tool"].split()[0] for i in elig if records[i].get("tool")}
    todo = [i for i in elig if not any(x["accepted"] for x in done.get(i, []))
            and len(done.get(i, [])) < a.attempts]
    print(f"{len(elig)} eligible, {len(todo)} to run, attempts {a.attempts}", flush=True)
    lock = threading.Lock()
    t0 = time.time()
    n = acc = 0
    with open(a.out, "a") as fh, concurrent.futures.ThreadPoolExecutor(a.concurrency) as ex:
        def job(i):
            res = None
            for att in range(len(done.get(i, [])), a.attempts):
                res = collect(a, i, records[i], docs, bins, att)
                with lock:
                    fh.write(json.dumps(res) + "\n")
                    fh.flush()
                if res["accepted"]:
                    break
            return res
        for fut in concurrent.futures.as_completed([ex.submit(job, i) for i in todo]):
            res = fut.result()
            n += 1
            acc += bool(res and res["accepted"])
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} done, {acc} accepted, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""G6t (plan.md "G6t"), step 2: research_dataset_g6q with the accepted base-Lightning
traces from probes/g6t_collect.py attached. A record with an accepted trace gains
"trace" (history as the eval harness sent it, base's completion text per turn, and
vLLM's token counts for the trainer's guard); g6_train.py RENDER=trace trains it one
sequence per turn. Only clean traces are attached: every call executed. A trace with a
refused call (base's `--help | grep` pipes) or a web_search (unavailable in training and
eval) would teach base's habit of spending its 3 calls, the call-limit failure base shows
in the eval, which G6q does not have. Every other record, and every field of every record, is G6q's.

usage: [LABEL=g6u] python3 probes/g6t_build.py TRACES.jsonl [TRACES.jsonl...]
writes results/research_dataset_<LABEL>.json and results/<LABEL>-build.json (the audit);
LABEL unset = g6t. With several trace files, a record takes its trace from the first
file that holds a clean accepted one (G6u: base's traces first, then G6t's).
PARTIAL=1 (G6u): a record with no clean accepted trace may take a clean attempt whose
only rejection is "no reasoning" (a turn closed </think> empty); its trace then carries
"train_turns", the turns with reasoning, and only those are trained (g6_train RENDER=trace
trains every turn when the key is absent).
"""
import collections
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G6Q = os.path.join(REPO, "results", "research_dataset_g6q.json")
LABEL = os.environ.get("LABEL") or "g6t"
OUT = os.path.join(REPO, "results", f"research_dataset_{LABEL}.json")
AUDIT = os.path.join(REPO, "results", f"{LABEL}-build.json")


def main():
    fail = []
    records = json.load(open(G6Q))
    tries = collections.defaultdict(list)
    accepted, source = {}, {}
    for path in sys.argv[1:]:
        mine = collections.defaultdict(list)
        for line in open(path):
            x = json.loads(line)
            mine[x["index"]].append(x)
        for i, xs in mine.items():
            tries[i].extend(xs)
            ok = [x for x in xs if x["accepted"] and all(c["outcome"] == "executed" for c in x["calls"])]
            if len(ok) > 1:
                fail.append(f"record {i}: {len(ok)} accepted attempts in {path}")
            if ok and i not in accepted:
                accepted[i], source[i] = ok[0], os.path.basename(path)
    partial = {}
    if os.environ.get("PARTIAL") == "1":
        for path in sys.argv[1:]:
            for line in open(path):
                x = json.loads(line)
                i = x["index"]
                if i in accepted or i in partial or not x["why"] or set(x["why"]) != {"no reasoning"}:
                    continue
                if not all(c["outcome"] == "executed" for c in x["calls"]):
                    continue
                turns = [k for k, tx in enumerate(x["texts"]) if "</think>" in tx and tx.split("</think>")[0].strip()]
                if turns:
                    partial[i] = dict(x, train_turns=turns)
                    source[i] = os.path.basename(path) + " (partial)"
        accepted.update(partial)
    out = json.loads(json.dumps(records))
    for i, x in accepted.items():
        r = out[i]
        if r.get("thinking") != "default" or r.get("type") != x["type"]:
            fail.append(f"record {i}: trace on a {r.get('thinking')} {r.get('type')} record")
        if x["messages"][0]["content"] != r["messages"][0]["content"]:
            fail.append(f"record {i}: trace prompt is not the record's")
        if len(x["texts"]) != len(x["calls"]) + 1 or len(x["messages"]) != 1 + 2 * len(x["calls"]):
            fail.append(f"record {i}: {len(x['texts'])} texts, {len(x['calls'])} calls, {len(x['messages'])} messages")
        if None in x.get("prompt_tokens", [None]):
            fail.append(f"record {i}: no vLLM prompt token counts")
        r["trace"] = {"index": i, "attempt": x["attempt"], **({"source": source[i]} if len(sys.argv) > 2 else {}),
                      **({"train_turns": x["train_turns"]} if "train_turns" in x else {}), "messages": x["messages"], "texts": x["texts"],
                      "prompt_tokens": x["prompt_tokens"], "completion_tokens": x["completion_tokens"]}
    for i, r in enumerate(out):
        if {k: v for k, v in r.items() if k != "trace"} != records[i]:
            fail.append(f"record {i}: G6q fields changed")
    by = collections.defaultdict(lambda: {"tried": 0, "accepted": 0, "accepted_clean": 0})
    for i, xs in tries.items():
        by[records[i]["type"]]["tried"] += 1
        by[records[i]["type"]]["accepted"] += any(x["accepted"] for x in xs)
        by[records[i]["type"]]["accepted_clean"] += i in accepted
    rc = [c for x in accepted.values() for c in x["reasoning_chars"]]
    audit = {
        "records": len(out), "tried": len(tries), "attempts_run": sum(len(v) for v in tries.values()),
        "accepted": len(accepted), "by_type": dict(sorted(by.items())),
        "by_source": dict(collections.Counter(source.values())),
        "accepted_by_attempt": dict(collections.Counter(x["attempt"] for x in accepted.values())),
        "trace_turns": sum(len(x["texts"]) for x in accepted.values()),
        "partial_records": len(partial),
        "partial_by_type": dict(collections.Counter(records[i]["type"] for i in partial)),
        "trained_trace_turns": sum(len(x.get("train_turns", x["texts"])) for x in accepted.values()),
        "calls_per_trace": dict(collections.Counter(len(x["calls"]) for x in accepted.values())),
        "tools_called": dict(collections.Counter(c["name"] for x in accepted.values() for c in x["calls"])),
        "reasoning_chars": {"min": min(rc), "median": statistics.median(rc), "max": max(rc)} if rc else None,
        "max_turn_tokens": max((t for x in accepted.values() for t in x["tokens"]), default=None),
        "default_records_without_trace": sum(r.get("thinking") == "default" and "trace" not in r for r in out),
        "rejections": dict(collections.Counter(w.split(" ref:")[0].split(" from ref")[0][:40]
                                               for xs in tries.values() for x in xs for w in x["why"]).most_common(15)),
        "failures": fail,
    }
    json.dump(audit, open(AUDIT, "w"), indent=1)
    print(json.dumps(audit, indent=1))
    if fail:
        sys.exit(f"ABORT: {len(fail)} failure(s); {OUT} not written")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {len(out)} records, {len(accepted)} with a trace -> {OUT}")


if __name__ == "__main__":
    main()

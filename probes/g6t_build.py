"""G6t (plan.md "G6t"), step 2: research_dataset_g6q with the accepted base-Lightning
traces from probes/g6t_collect.py attached. A record with an accepted trace gains
"trace" (history as the eval harness sent it, base's completion text per turn, and
vLLM's token counts for the trainer's guard); g6_train.py RENDER=trace trains it one
sequence per turn. Only clean traces are attached: every call executed. A trace with a
refused call (base's `--help | grep` pipes) or a web_search (unavailable in training and
eval) would teach base's habit of spending its 3 calls, the call-limit failure base shows
in the eval, which G6q does not have. Every other record, and every field of every record, is G6q's.

usage: python3 probes/g6t_build.py results/g6t-traces.jsonl
writes results/research_dataset_g6t.json and results/g6t-build.json (the audit).
"""
import collections
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G6Q = os.path.join(REPO, "results", "research_dataset_g6q.json")
OUT = os.path.join(REPO, "results", "research_dataset_g6t.json")
AUDIT = os.path.join(REPO, "results", "g6t-build.json")


def main():
    fail = []
    records = json.load(open(G6Q))
    tries = collections.defaultdict(list)
    for line in open(sys.argv[1]):
        x = json.loads(line)
        tries[x["index"]].append(x)
    out = json.loads(json.dumps(records))
    accepted = {}
    for i, xs in tries.items():
        ok = [x for x in xs if x["accepted"] and all(c["outcome"] == "executed" for c in x["calls"])]
        if len(ok) > 1:
            fail.append(f"record {i}: {len(ok)} accepted attempts")
        if ok:
            accepted[i] = ok[0]
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
        r["trace"] = {"index": i, "attempt": x["attempt"], "messages": x["messages"], "texts": x["texts"],
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
        "accepted_by_attempt": dict(collections.Counter(x["attempt"] for x in accepted.values())),
        "trace_turns": sum(len(x["texts"]) for x in accepted.values()),
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

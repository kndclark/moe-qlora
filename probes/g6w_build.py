"""G6w (plan.md "G6w"): research_dataset_g6u unchanged except the 60 web_research records,
whose web_search call now fails and whose answer says so.

Why: G6u, thinking on, calls web_search on 72 of 123 trap items (N1's three repeats
pooled), and 27 of its 40 trap failures follow one: 11 write the denial inside an unclosed
<think>, 11 fall back to "I couldn't check the documentation" although the lookup ran, 5 hit
the call limit. G6q (same web_research records, no traces) calls it on none. The eval's
web_search always answers "unavailable", and no lab service implements it; yet in the
training data it succeeds 60 times out of 108 (6 questions x 10, each answered from the
returned snippet) and fails only in web_fallback's 48, which then run --help. The data
never shows what to do when a search fails and no lookup follows.

Change: each web_research record's tool result becomes one of augment_research.WEB_FAILURES
(seeded; never the eval's own string, as augment_research requires), and its answer keeps
its content but drops the claimed source: "I couldn't search the documentation (<why>),
so this is from memory and unverified: <content> Check <where> before relying on it.",
the shape lookup_failed already uses. Question, call, thinking mode and order unchanged.
After it, every web_search result in the dataset is a failure.

usage: python3 probes/g6w_build.py
writes results/research_dataset_g6w.json and results/g6w-build.json (the audit).
"""
import collections
import hashlib
import json
import os
import random
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/gpu-lab/training"))
from augment_research import WEB_FAILURES  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G6U = os.path.join(REPO, "results", "research_dataset_g6u.json")
OUT = os.path.join(REPO, "results", "research_dataset_g6w.json")
AUDIT = os.path.join(REPO, "results", "g6w-build.json")
SEED = 20260929
EVAL_WEB = "web_search is unavailable in this evaluation."

WHY = {
    "Error: search backend returned HTTP 503 (service unavailable)": "the search backend returned HTTP 503",
    "web_search failed: network is unreachable": "the network is unreachable",
    "Search request timed out after 20 s; no results.": "the search timed out",
}
# answer opening (the claimed source, dropped) -> where to check
SOURCES = {
    "Based on the PyTorch tensor documentation, ":
        "the PyTorch documentation for `Tensor.to`",
    "Based on online documentation for `sqlx::sqlite::SqliteConnectOptions`, ":
        "the sqlx documentation for `SqliteConnectOptions`",
    "According to NVIDIA CUDA Compiler Driver NVCC documentation, ":
        "`nvcc --help` or NVIDIA's NVCC documentation",
    "Based on the `vllm.EngineArgs` specification, ":
        "vLLM's `EngineArgs` documentation",
    "According to tokio documentation for `tokio::runtime::Builder`, ":
        "the tokio documentation for `runtime::Builder`",
    "According to Hugging Face PEFT documentation, ":
        "the PEFT documentation for `LoraConfig`",
}


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def web_results(recs):
    out = []
    for r in recs:
        msgs = r["messages"]
        for i, m in enumerate(msgs):
            for tc in m.get("tool_calls") or []:
                if tc["function"]["name"] == "web_search":
                    out.append(msgs[i + 1]["content"])
    return out


def main():
    assert set(WHY) == set(WEB_FAILURES), "augment_research.WEB_FAILURES changed"
    g6u = json.load(open(G6U))
    rng = random.Random(SEED)
    out, changed, fails, by_q = [], [], collections.Counter(), collections.Counter()
    for n, r in enumerate(g6u):
        if r["type"] != "web_research":
            out.append(r)
            continue
        r = json.loads(json.dumps(r))
        user, call, tool, answer = r["messages"]
        assert (user["role"], call["role"], tool["role"], answer["role"]) == \
            ("user", "assistant", "tool", "assistant"), n
        assert call["tool_calls"][0]["function"]["name"] == "web_search", n
        assert "trace" not in r, n  # a trace would carry reasoning about the old result
        opener = next((o for o in SOURCES if answer["content"].startswith(o)), None)
        assert opener, f"record {n}: unknown answer opening {answer['content'][:60]!r}"
        fail = rng.choice(WEB_FAILURES)
        tool["content"] = fail
        answer["content"] = (f"I couldn't search the documentation ({WHY[fail]}), so this is from "
                             f"memory and unverified: {answer['content'][len(opener):]} "
                             f"Check {SOURCES[opener]} before relying on it.")
        out.append(r)
        changed.append(n)
        fails[fail] += 1
        by_q[user["content"]] += 1

    others_same = all(a == b for a, b in zip(g6u, out) if a["type"] != "web_research")
    results = web_results(out)
    audit = {
        "input": os.path.basename(G6U), "input_sha": sha(g6u), "output_sha": sha(out),
        "records": len(out), "changed": len(changed), "changed_indices": changed,
        "others_identical": others_same and len(out) == len(g6u),
        "failure_strings": dict(fails), "questions": dict(by_q),
        "thinking": dict(collections.Counter(out[n].get("thinking") for n in changed)),
        "web_search_results_before": len(web_results(g6u)),
        "web_search_successes_before": sum(x not in WHY for x in web_results(g6u)),
        "web_search_results_after": len(results),
        "web_search_successes_after": sum(x not in WHY for x in results),
        "eval_string_in_data": sum(EVAL_WEB in json.dumps(r) for r in out),
        "sample": out[changed[0]]["messages"][2:],
    }
    fail = [k for k, ok in (("others_identical", audit["others_identical"]),
                            ("changed == 60", len(changed) == 60),
                            ("no web_search success left", audit["web_search_successes_after"] == 0),
                            ("eval string absent", audit["eval_string_in_data"] == 0)) if not ok]
    json.dump(audit, open(AUDIT, "w"), indent=1)
    print(json.dumps({k: v for k, v in audit.items() if k != "changed_indices"}, indent=1))
    if fail:
        sys.exit(f"\nABORT: {fail}; {OUT} not written")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {len(out)} records ({len(changed)} web_research rewritten) -> {OUT}")


if __name__ == "__main__":
    main()

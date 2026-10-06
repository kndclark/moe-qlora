"""Greedy outputs and speed from a running server, for comparing kstage modes (stdlib only).
decode: streams of one "Story" prompt (they route alike, which flatters an expert cache);
decode_mixed, measured last: each stream a different task (code, prose, math, SQL, ...).
usage: kstage_client.py BASE MODEL OUT.json"""
import json, random, sys, threading, time, urllib.request

base, model, out = sys.argv[1:4]
PROMPTS = [
    "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n",
    "The three main causes of the French Revolution were",
    "Q: A train leaves at 9:40 and arrives at 13:05. How long is the trip?\nA:",
    "#!/bin/bash\n# Rotate logs older than 7 days in /var/log/app and compress them\n",
    "Explain, step by step, why the sky is blue.",
    "SELECT customer_id, SUM(amount) AS total\nFROM orders\n",
    "Translate to French: The meeting has been moved to Thursday afternoon.",
    "Write a haiku about a GPU running out of memory.",
]


def post(path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def complete(prompt, n, **kw):
    return post("/v1/completions", {"model": model, "prompt": prompt, "max_tokens": n, "temperature": 0,
                                    "return_token_ids": True, **kw})["choices"][0]


res = {"greedy": []}
for p in PROMPTS:
    c = complete(p, 96, logprobs=5)   # the first token's top 5 tell a near-tie from a wrong prefill
    tops = (c.get("logprobs") or {}).get("top_logprobs") or [None]
    res["greedy"].append({"text": c["text"], "ids": c.get("token_ids"), "top0": tops[0],
                          # each position's top 2: a later divergence at a near-tie is not a fault
                          "top2": [sorted(t.values(), reverse=True)[:2] if t else None for t in tops]})


MIXED = PROMPTS + [
    "// Rust: a thread-safe LRU cache with a fixed capacity\nuse std::collections::HashMap;\n",
    "Prove that the square root of 2 is irrational.",
    "Dear hiring manager,\nI am writing to apply for the position of",
    "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: web\nspec:\n",
    "List the planets of the solar system with one fact about each:\n1.",
    "Summarize the plot of Hamlet in five sentences.",
    "import numpy as np\n\ndef softmax(x, axis=-1):\n",
    "A recipe for bread with only flour, water, salt and yeast:\n",
]


def decode(conc, n=256, mixed=False):
    outs, ths = [None] * conc, []
    def one(i):
        p = MIXED[i % len(MIXED)] if mixed else f"Story {i}: Once upon a time"
        outs[i] = complete(p, n, ignore_eos=True)
    t0 = time.perf_counter()
    for i in range(conc):
        ths.append(threading.Thread(target=one, args=(i,))); ths[-1].start()
    for t in ths:
        t.join()
    return conc * n / (time.perf_counter() - t0)


complete("warm up", 8)
res["decode"] = {str(c): round(decode(c), 1) for c in (1, 4, 16)}
json.dump(res, open(out, "w"), indent=1)
rng = random.Random(7)
words = [str(rng.randrange(10**6)) for _ in range(1100)]  # unique, so no prefix hit; digits tokenize singly
try:
    t0 = time.perf_counter(); u = post("/v1/completions", {"model": model, "prompt": " ".join(words), "max_tokens": 1})
    res["prefill_s"], res["prefill_tokens"] = round(time.perf_counter() - t0, 2), u["usage"]["prompt_tokens"]
except Exception as e:  # noqa: BLE001
    res["prefill_s"], res["prefill_error"] = None, str(e)
json.dump(res, open(out, "w"), indent=1)
res["decode_mixed"] = {str(c): round(decode(c, mixed=True), 1) for c in (1, 4, 16)}
json.dump(res, open(out, "w"), indent=1)
print(f"decode tok/s {res['decode']}, mixed {res['decode_mixed']}; prefill {res.get('prefill_tokens')} tokens in "
      f"{res['prefill_s']} s {res.get('prefill_error', '')}; greedy saved to {out}")

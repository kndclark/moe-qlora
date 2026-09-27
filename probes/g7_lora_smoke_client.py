"""G7 smoke, part 3: compare base Lightning with each part-1 adapter, through vLLM.

usage: g7_lora_smoke_client.py BASE_URL OUT_JSON

Greedy completions (16 tokens, logprobs) of three prompts per model, sent one at
a time. Base runs twice: that difference is the noise floor. Per adapter:
  loads    = vLLM served it at all
  changed  = any token differs from base, or a logprob on the shared prefix
             moves by more than max(10 x noise, 1e-3)
Expected: zero unchanged (it loads and is a no-op); attn, inproj, shexp and full
changed. An adapter family that comes back unchanged was skipped by vLLM.
"""
import json
import sys
import urllib.request

BASE, OUT = sys.argv[1], sys.argv[2]
PROMPTS = ["The capital of France is",
           "def fibonacci(n):\n    ",
           "In one sentence, a state-space model differs from attention in that"]
ADAPTERS = ["zero", "attn", "inproj", "shexp", "full"]
EXPECT_CHANGED = {"zero": False, "attn": True, "inproj": True, "shexp": True, "full": True}


def complete(model):
    runs = []
    for p in PROMPTS:
        body = json.dumps({"model": model, "prompt": p, "max_tokens": 16, "temperature": 0,
                           "logprobs": 1}).encode()
        req = urllib.request.Request(BASE + "/v1/completions", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            c = json.load(r)["choices"][0]
        runs.append({"text": c["text"], "tokens": c["logprobs"]["tokens"],
                     "logprobs": c["logprobs"]["token_logprobs"]})
    return runs


def compare(a, b):
    same = all(x["tokens"] == y["tokens"] for x, y in zip(a, b))
    dmax = 0.0
    for x, y in zip(a, b):
        for tx, ty, lx, ly in zip(x["tokens"], y["tokens"], x["logprobs"], y["logprobs"]):
            if tx != ty:
                break
            dmax = max(dmax, abs(lx - ly))
    return same, dmax


base = complete("lightning-nvfp4")
base2 = complete("lightning-nvfp4")
same, noise = compare(base, base2)
report = {"prompts": PROMPTS, "base": base, "noise": {"same_tokens": same, "max_dlogprob": noise}}
threshold = max(10 * noise, 1e-3)
ok = same
for name in ADAPTERS:
    try:
        runs = complete(name)
    except Exception as e:  # recorded, not raised: one family failing must not hide the rest
        report[name] = {"loads": False, "error": repr(e)[:400]}
        ok = False
        continue
    s, d = compare(base, runs)
    changed = (not s) or d > threshold
    report[name] = {"loads": True, "same_tokens": s, "max_dlogprob": d, "changed": changed,
                    "expected_changed": EXPECT_CHANGED[name],
                    "pass": changed == EXPECT_CHANGED[name], "runs": runs}
    ok = ok and changed == EXPECT_CHANGED[name]
report["threshold"] = threshold
report["pass"] = ok
with open(OUT, "w") as f:
    json.dump(report, f, indent=1)
print(f"noise: same_tokens={same} max_dlogprob={noise:.2e}  threshold={threshold:.2e}")
for name in ADAPTERS:
    r = report[name]
    if not r["loads"]:
        print(f"  {name:7s} FAILED TO SERVE: {r['error'][:160]}")
    else:
        print(f"  {name:7s} same_tokens={r['same_tokens']!s:5s} max_dlogprob={r['max_dlogprob']:.2e} "
              f"changed={r['changed']!s:5s} expected={r['expected_changed']!s:5s} "
              f"{'PASS' if r['pass'] else 'FAIL'}")
print("PASS" if ok else "FAIL")

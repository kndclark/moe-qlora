"""Workloads for the expert-offload program (docs/laptop-memory-levers.md, "Experts in RAM").

The short eval only measures what offload costs: its 16 requests fit without it. These
measure what it is for, long-context agent serving, against a running server:
  decode   C parallel streams, distinct 1k-token prompts, G tokens each (ignore_eos):
           per-stream and aggregate decode tok/s, for each C in --conc
  prefill  one request per length in --lens, max_tokens 1: cold time-to-first-token and
           prefill tok/s, then the same prompt again (warm: prefix-cache hit)
  agent    N agents in parallel, each a loop: context grows by --chunk new tokens a turn
           (a tool result), G tokens are generated, until --target tokens. Per turn:
           prompt tokens, cached tokens, TTFT, decode tok/s. This is the target workload.
  needle   long-context recall: K "the access code for <name> is <6 digits>" lines spread
           through real text, ask for each code; exact-match score per length in --lens
  experts  routed-expert histogram (server needs --enable-return-routed-experts): runs the
           research-eval prompts and agent-like text, saves per-layer counts per expert
Prompts are real local text (Python's standard library source), tokenized by the server
and sent as token ids, so lengths are exact and no two streams share a prefix by accident.
usage: offload_bench.py --base URL --model NAME --out FILE.json MODE [MODE...] [options]
"""
import argparse
import base64
import glob
import http.client
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


def post(base, path, body, timeout=3600):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:  # the server's reason, not just "400 Bad Request"
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from None


def metrics(base):
    want = ("vllm:num_preemptions_total", "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
            "vllm:external_prefix_cache_hits_total", "vllm:external_prefix_cache_queries_total")
    out = {}
    with urllib.request.urlopen(base + "/metrics", timeout=30) as r:
        for line in r.read().decode().splitlines():
            for w in want:
                if line.startswith(w + "{") or line.startswith(w + " "):
                    out[w] = out.get(w, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def corpus_tokens(base, model, need):
    """Token ids of real text, at least `need` of them, deterministic."""
    files = sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))
    ids, chunk = [], []
    for f in files:
        try:
            chunk.append(open(f, errors="replace").read())
        except OSError:
            continue
        if sum(map(len, chunk)) > 400_000:
            ids += post(base, "/tokenize", {"model": model, "prompt": "".join(chunk), "add_special_tokens": False})["tokens"]
            chunk = []
            if len(ids) >= need:
                return ids
    if chunk:
        ids += post(base, "/tokenize", {"model": model, "prompt": "".join(chunk), "add_special_tokens": False})["tokens"]
    if len(ids) < need:
        sys.exit(f"corpus has {len(ids)} tokens, need {need}")
    return ids


def stream(base, model, prompt, gen, extra=None):
    """One streamed completion. Returns ttft, decode tok/s, usage, completion tokens."""
    u = urllib.parse.urlparse(base)
    body = {"model": model, "prompt": prompt, "max_tokens": gen, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    body.update(extra or {})
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=7200)
    t0 = time.time()
    c.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        return {"error": f"{r.status} {r.read()[:300].decode(errors='replace')}"}
    first = last = None
    n, usage = 0, None
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            usage = d["usage"]
        if d.get("choices") and d["choices"][0].get("text") is not None:
            now = time.time()
            first = first or now
            last = now
            n += 1
    c.close()
    if first is None:
        return {"error": "no tokens", "usage": usage}
    toks = (usage or {}).get("completion_tokens", n)
    dt = last - first
    return {"ttft": first - t0, "decode_tps": (toks - 1) / dt if dt > 0 and toks > 1 else None,
            "total_s": last - t0, "usage": usage}


def parallel(fns):
    out = [None] * len(fns)

    def run(i, f):
        out[i] = f()
    ts = [threading.Thread(target=run, args=(i, f)) for i, f in enumerate(fns)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out, time.time() - t0


def mode_decode(a, ids):
    res, off = [], 0
    for c in a.conc:
        fns = []
        for _ in range(c):
            p = ids[off:off + a.prompt]
            off += a.prompt
            fns.append(lambda p=p: stream(a.base, a.model, p, a.gen))
        out, wall = parallel(fns)
        ok = [o for o in out if "error" not in o]
        per = sorted(o["decode_tps"] for o in ok if o["decode_tps"])
        agg = sum(o["usage"]["completion_tokens"] for o in ok) / max(max(o["total_s"] for o in ok) - min(o["ttft"] for o in ok), 1e-9) if ok else 0
        r = {"conc": c, "ok": len(ok), "per_stream_tps_median": per[len(per) // 2] if per else None,
             "per_stream_tps_min": per[0] if per else None, "aggregate_tps": agg, "wall_s": wall,
             "errors": [o["error"] for o in out if "error" in o][:2]}
        print(f"  decode c={c}: per-stream median {r['per_stream_tps_median'] or 0:.1f} tok/s, "
              f"aggregate {agg:.1f} tok/s, {len(ok)}/{c} ok", flush=True)
        res.append(r)
    return res, off


def mode_prefill(a, ids, off):
    res = []
    for n in a.lens:
        p = ids[off:off + n]
        off += n
        cold = stream(a.base, a.model, p, 2)
        warm = stream(a.base, a.model, p, 2) if "error" not in cold else {"error": "skipped"}
        r = {"len": n, "cold_ttft": cold.get("ttft"), "warm_ttft": warm.get("ttft"),
             "prefill_tps": n / cold["ttft"] if cold.get("ttft") else None,
             "cached_warm": ((warm.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
             "errors": [x["error"] for x in (cold, warm) if "error" in x]}
        print(f"  prefill {n}: cold {r['cold_ttft'] or 0:.2f}s ({r['prefill_tps'] or 0:.0f} tok/s), "
              f"warm {r['warm_ttft'] or 0:.2f}s, cached {r['cached_warm']} {r['errors'][:1]}", flush=True)
        res.append(r)
    return res, off


def mode_agent(a, ids, off):
    per_agent = a.start + a.target  # generous slice per agent
    m0 = metrics(a.base)
    t0 = time.time()

    def agent(k):
        src = ids[off + k * per_agent: off + (k + 1) * per_agent]
        ctx, pos, turns = list(src[:a.start]), a.start, []
        while len(ctx) + a.gen <= a.target:
            o = stream(a.base, a.model, ctx, a.gen)
            u = o.get("usage") or {}
            turns.append({"prompt": len(ctx), "ttft": o.get("ttft"), "decode_tps": o.get("decode_tps"),
                          "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                          "error": o.get("error"), "t": time.time() - t0})
            if o.get("error"):
                break
            ctx += src[pos:pos + a.gen + a.chunk]  # stand-in for the reply, then the next tool result
            pos += a.gen + a.chunk
        return turns
    out, wall = parallel([lambda k=k: agent(k) for k in range(a.agents)])
    m1 = metrics(a.base)
    delta = {k.split(":")[1]: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    allt = [t for ts in out for t in ts if not t["error"]]
    dec = sorted(t["decode_tps"] for t in allt if t["decode_tps"])
    big = [t for t in allt if t["prompt"] >= a.target // 2]
    r = {"agents": a.agents, "wall_s": wall, "turns": sum(map(len, out)), "errors": sum(1 for ts in out for t in ts if t["error"]),
         "decode_tps_median": dec[len(dec) // 2] if dec else None,
         "ttft_median_second_half": sorted(t["ttft"] for t in big)[len(big) // 2] if big else None,
         "metrics_delta": delta, "per_agent": out}
    print(f"  agent x{a.agents} to {a.target}: {r['turns']} turns, {r['errors']} errors, wall {wall:.0f}s, decode median "
          f"{r['decode_tps_median'] or 0:.1f} tok/s, late TTFT median {r['ttft_median_second_half'] or 0:.2f}s, "
          f"preemptions {delta.get('num_preemptions_total', 0):.0f}, prefix hits/queries "
          f"{delta.get('prefix_cache_hits_total', 0):.0f}/{delta.get('prefix_cache_queries_total', 0):.0f}, "
          f"external hits {delta.get('external_prefix_cache_hits_total', 0):.0f}", flush=True)
    return r


def mode_needle(a):
    files = sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))
    text = []
    for f in files:
        try:
            text.append(open(f, errors="replace").read())
        except OSError:
            pass
        if sum(map(len, text)) > 6_000_000:
            break
    text = "".join(text)
    rng = random.Random(20261005)
    names = ["Arden", "Bexley", "Corin", "Dalby", "Elstow", "Farrow", "Garnet", "Hollis"]
    # Python source is full of NAME = NUMBER constants, so the needles are named and quoted
    # exactly, before and after the file (round 8's open question got constants back).
    pre = ("Below is a long file. Eight comment lines hidden in it read exactly "
           "'# NOTE: the access code for NAME is NUMBER.', for the names " + ", ".join(names) + ".\n\n")
    q = ("\n\nIn the file above, find the eight '# NOTE: the access code for NAME is NUMBER.' comment lines. "
         "Write each name (" + ", ".join(names) + ") with its six-digit code, one per line as NAME: NUMBER, "
         "and nothing else.")
    kw = {"chat_template_kwargs": {"enable_thinking": False}}

    def build(chars, codes, fr):
        hay, parts, prev = text[:chars], [], 0
        for f, (nm, c) in zip(fr, codes.items()):
            cut = int(f * chars)
            cut = hay.rfind("\n", prev, cut) if hay.rfind("\n", prev, cut) > prev else cut
            parts += [hay[prev:cut], f"\n# NOTE: the access code for {nm} is {c}.\n"]
            prev = cut
        parts.append(hay[prev:])
        return [{"role": "user", "content": pre + "".join(parts) + q}]

    def count(msgs):  # exact prompt tokens, chat template included
        r = post(a.base, "/tokenize", {"model": a.model, "messages": msgs, "add_generation_prompt": True, **kw})
        return r["count"], r.get("max_model_len")

    res = []
    for n in a.lens:
        codes = {nm: f"{rng.randrange(10**5, 10**6)}" for nm in names}
        fr = sorted(0.05 + 0.9 * rng.random() for _ in names)
        chars, msgs, got_n, cap = int(n * 3.0), None, 0, None
        for _ in range(4):  # chars per token varies ~4% across the text: measure, rescale, measure
            if chars > len(text) * 0.95:
                sys.exit(f"needle text has {len(text)} chars, {n} tokens needs {chars}")
            msgs = build(chars, codes, fr)
            got_n, cap = count(msgs)
            limit = min(n, (cap or n) - 200)  # leave room for the answer
            if 0.98 * limit <= got_n <= limit:
                break
            chars = int(chars * limit * 0.99 / got_n)
        t0 = time.time()
        try:
            if got_n > (cap or got_n) - 200:
                raise RuntimeError(f"prompt {got_n} tokens does not fit max_model_len {cap}")
            o = post(a.base, "/v1/chat/completions", {"model": a.model, "temperature": 0, "max_tokens": 200,
                     "messages": msgs, **kw})
            ans = o["choices"][0]["message"]["content"] or ""
            got = sum(1 for nm, c in codes.items() if f"{nm}: {c}" in ans or (nm in ans and c in ans.split(nm, 1)[1][:20]))
            r = {"len_target": n, "prompt_tokens": o["usage"]["prompt_tokens"], "found": got, "of": len(codes),
                 "seconds": time.time() - t0, "answer": ans[:400]}
        except Exception as e:  # noqa: BLE001 - report and continue
            r = {"len_target": n, "error": str(e)[:300]}
        print(f"  needle {n}: {r.get('found')}/{r.get('of')} at {r.get('prompt_tokens')} tokens, "
              f"{r.get('seconds', 0):.0f}s {r.get('error', '')}", flush=True)
        res.append(r)
    return res


def npy_u1(b):
    """(shape, raw bytes) of a C-order uint8 .npy payload, without numpy."""
    import ast
    if b[:6] != b"\x93NUMPY":
        sys.exit("routed_experts is not an .npy payload")
    hl, start = (int.from_bytes(b[8:10], "little"), 10) if b[6] == 1 else (int.from_bytes(b[8:12], "little"), 12)
    h = ast.literal_eval(b[start:start + hl].decode())
    if h["descr"] not in ("|u1", "<u1") or h["fortran_order"]:
        sys.exit(f"unexpected routed_experts layout {h}")
    return h["shape"], b[start + hl:]


def mode_experts(a, ids, off):
    """Per-layer routed-expert counts over eval prompts and code text; raw routings saved
    to OUT.experts-KIND.bin ([tokens, layers, top_k] uint8, request after request; "lens"
    holds each request's [routed tokens, prompt tokens], to cut the file back into requests)."""
    from collections import Counter
    prompts = []
    for s in ("v1", "v2", "alert", "trap3"):
        try:
            d = json.load(open(a.eval_json.format(s)))
            prompts += [r["question"] + "\n\n" + "\n".join(t.get("text") or "" for t in (r.get("run") or {}).get("turns", []))
                        for r in d["results"]]  # no prompts are stored: the question plus G6q's own replies
        except (OSError, KeyError, ValueError):
            pass
    jobs = [("eval", p) for p in prompts[: a.max_prompts]]
    jobs += [("code", ids[off + i * 4096: off + (i + 1) * 4096]) for i in range(8)]
    counts, ntok, files, shape, lens = {}, {}, {}, None, {}
    for kind, p in jobs:
        try:
            o = post(a.base, "/v1/completions", {"model": a.model, "prompt": p, "max_tokens": a.gen, "temperature": 0})
        except Exception as e:  # noqa: BLE001
            print("  experts: request failed", str(e)[:200]); continue
        b = o["choices"][0].get("routed_experts")
        if not b:
            sys.exit("no routed_experts in response: start the server with --enable-return-routed-experts")
        shape, raw = npy_u1(base64.b64decode(b))
        n, L, K = shape
        c = counts.setdefault(kind, [Counter() for _ in range(L)])
        for layer in range(L):
            for k in range(K):
                c[layer].update(raw[layer * K + k::L * K])
        ntok[kind] = ntok.get(kind, 0) + n
        lens.setdefault(kind, []).append([n, o["usage"]["prompt_tokens"]])
        if kind not in files:
            files[kind] = open(f"{a.out}.experts-{kind}.bin", "wb")
        files[kind].write(raw)
    for f in files.values():
        f.close()
    out = {"shape_layers_topk": shape[1:] if shape else None, "tokens": ntok, "lens": lens,
           "counts": {k: [[v[e] for e in range(a.num_experts)] for v in c] for k, c in counts.items()}}
    for k, c in out["counts"].items():
        tops = []
        for row in c:
            if sum(1 for v in row if v) <= (shape[2] if shape else 6):
                continue  # a Mamba or attention layer: vLLM returns a row for it, with no routing
            srt, tot = sorted(row, reverse=True), sum(row)
            tops.append([sum(srt[:m]) / tot for m in (13, 32, 64)])
        med = [sorted(t[i] for t in tops)[len(tops) // 2] for i in range(3)]
        print(f"  experts {k}: {ntok[k]} tokens; share of routings to each layer's top 13/32/64 experts "
              f"(median of {len(tops)} MoE layers): {med[0]:.2f}/{med[1]:.2f}/{med[2]:.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modes", nargs="+", choices=("decode", "prefill", "agent", "needle", "experts"))
    ap.add_argument("--base", default="http://127.0.0.1:8303")
    ap.add_argument("--model", default="g6q")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", type=lambda s: [int(x) for x in s.split(",")], default=[1, 2, 4, 8, 16])
    ap.add_argument("--prompt", type=int, default=1024)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--lens", type=lambda s: [int(x) for x in s.split(",")], default=[8192, 32768])
    ap.add_argument("--agents", type=int, default=2)
    ap.add_argument("--start", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--target", type=int, default=65536)
    ap.add_argument("--num-experts", type=int, default=128)
    ap.add_argument("--max-prompts", type=int, default=120)
    ap.add_argument("--eval-json", default="results/kv-levers/research-eval-{}-g6q-tm-graphs092.json")
    a = ap.parse_args()
    need = 16 * 1024 * 40 + sum(a.lens) * 2 + a.agents * (a.start + a.target) + 40000
    ids = corpus_tokens(a.base, a.model, need) if set(a.modes) - {"needle"} else []
    out, off = {"args": vars(a), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, 0
    for m in a.modes:
        if m == "decode":
            out["decode"], off = mode_decode(a, ids)
            off = 16 * 1024 * 40
        elif m == "prefill":
            out["prefill"], off = mode_prefill(a, ids, max(off, 16 * 1024 * 40))
        elif m == "agent":
            out["agent"] = mode_agent(a, ids, max(off, 16 * 1024 * 40 + sum(a.lens) * 2))
        elif m == "needle":
            out["needle"] = mode_needle(a)
        elif m == "experts":
            out["experts"] = mode_experts(a, ids, 16 * 1024 * 40)
        json.dump(out, open(a.out, "w"), indent=1)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

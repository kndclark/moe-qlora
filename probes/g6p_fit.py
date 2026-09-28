"""G6p (plan.md "G6p"): does every new record fit in 1024 tokens with its whole answer?

Renders with g6_train.py's own code, not a copy: this runs g6_train.py's source up to the
line that encodes the data (tokenizer, TOOLS, DATASET, lightning_messages, encode), then
encodes every record with RENDER as g6_train would. The full length is the same encode
with the length cap lifted. Records of type task_procedure are the new ones.

Checks, printed and written to /out/g6p-fit.json:
  - new records: full length min / median / max, and how many exceed MAX_LEN (must be 0);
    every assistant turn trained, the last trained token <|im_end|>;
  - all records: how many exceed MAX_LEN (v3 alone: 439) and the token total after the
    cap, which must equal the dry run's "tokens" for the same file (same render);
  - the old (v3) records by tool: how many keep their final answer whole, in part, or
    not at all under the cap, i.e. which answers G6 and G6r ever trained on.

Run (laptop, tokenizer only):
  DATASET=/out/research_dataset_g6p.json RENDER=think probes/gpurun.sh g6p-fit /probes/g6p_fit.py
"""
import json
import os
import statistics
import sys

src_path = "/probes/g6_train.py"
src = open(src_path).read()
marker = "\ndata = [encode(r, RENDER) for r in records]\n"
assert src.count(marker) == 1, "g6_train.py changed: encode marker not found once"
g = {"__name__": "g6_train_prefix", "__file__": src_path}
sys.argv = [src_path, "g6p-fit"]
exec(compile(src.split(marker)[0], src_path, "exec"), g)

encode, records, tok, RENDER = g["encode"], g["records"], g["tok"], g["RENDER"]
MAX_LEN, im_end = g["MAX_LEN"], g["im_end"]
capped = [encode(r, RENDER) for r in records]
g["MAX_LEN"] = 10 ** 9  # encode reads it from its module globals
full = [encode(r, RENDER) for r in records]
g["MAX_LEN"] = MAX_LEN

new = [i for i, r in enumerate(records) if r.get("type") == "task_procedure"]
lens = [len(full[i]["ids"]) for i in new]
bad = []
for i in new:
    d = capped[i]
    n_asst = sum(m["role"] == "assistant" for m in records[i]["messages"])
    trained = [j for j, l in enumerate(d["labels"]) if l != -100]
    if len(full[i]["ids"]) > MAX_LEN or len(d["turns"]) != n_asst or not trained or d["ids"][trained[-1]] != im_end:
        bad.append(i)
answer_tokens = []
for i in new:
    d = capped[i]
    h, start = d["turns"][-1]
    answer_tokens.append(sum(1 for l in d["labels"][start:] if l != -100))



def answer_kept(i):
    """How much of the record's final answer survives the cap: full, partial or none."""
    d = capped[i]
    n_asst = sum(m["role"] == "assistant" for m in records[i]["messages"])
    if len(d["turns"]) < n_asst or d["turns"][-1][1] >= len(d["ids"]):
        return "none"
    trained = [j for j, l in enumerate(d["labels"]) if l != -100]
    return "full" if d["ids"][trained[-1]] == im_end and len(full[i]["ids"]) <= MAX_LEN else "partial"


# The old records by tool (or type, for tool-free ones): whose answers training ever saw.
by_tool = {}
for i, r in enumerate(records):
    if r.get("type") == "task_procedure":
        continue
    k = r.get("tool") or f"({r.get('type')})"
    s = by_tool.setdefault(k, {"records": 0, "full": 0, "partial": 0, "none": 0})
    s["records"] += 1
    s[answer_kept(i)] += 1
res = {"dataset": g["DATASET"], "render": RENDER, "max_len": MAX_LEN, "records": len(records),
       "new_records": len(new),
       "new_full_len": {"min": min(lens), "median": statistics.median(lens), "max": max(lens)},
       "new_over_max_len": sum(n > MAX_LEN for n in lens),
       "new_answer_tokens": {"min": min(answer_tokens), "median": statistics.median(answer_tokens),
                             "max": max(answer_tokens)},
       "new_failing_records": bad,
       "all_over_max_len": sum(d["truncated"] for d in capped),
       "all_tokens_after_cap": sum(len(d["ids"]) for d in capped),
       "old_answers_kept": {k: sum(s[k] for s in by_tool.values()) for k in ("full", "partial", "none")},
       "old_by_tool": dict(sorted(by_tool.items(), key=lambda kv: (-kv[1]["none"], kv[0]))),
       "pass": not bad and len(new) > 0}
with open("/out/g6p-fit.json", "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "old_by_tool"}, indent=1))
print("old records by tool: records, final answer kept full / partial / none")
for k, s in res["old_by_tool"].items():
    print(f"  {k:28s} {s['records']:4d}  {s['full']:4d} {s['partial']:4d} {s['none']:4d}")
print("PASS" if res["pass"] else "FAIL")

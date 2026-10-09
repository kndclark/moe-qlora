"""G7a (plan.md): ~/gpu-lab/bench/research_eval.py, unchanged, on Nemotron 3.5 Lightning.
Two patches, both in the harness's reading of the model's text; the scorer is untouched:

  parse_tool_call  also reads the Qwen3-Coder XML form Lightning's template asks for
                   (<tool_call><function=NAME><parameter=K>V</parameter></function>), after
                   the harness's own Qwen-JSON and Llama forms have failed to match,
                   and GLM-4.5/4.7's form (<tool_call>NAME<arg_key>K</arg_key>
                   <arg_value>V</arg_value></tool_call>) after that. Values stay strings,
                   as the template writes them raw; every harness tool takes strings.
  split_think      with thinking on, Lightning's generation prompt ends in an open
                   "<think>\\n", so the completion carries "</think>" but never "<think>".
                   No "</think>" then means truncated mid-think, not an answer.

Tool arguments go back to the template as a mapping without help: vLLM 0.29 json-loads
assistant tool_call arguments before rendering (entrypoints/chat_utils.py:2043).

Two options of this wrapper (S1, docs/next-model-plan.md), both applied to the chat
request before /tokenize renders it, and recorded in the output JSON under "wrapper":

  --think-tag         for templates that switch thinking by a tag, not enable_thinking
                      (Nemotron Nano 9B v2): a system message "/think" or "/no_think"
                      per --thinking. The template strips the tag from the system text.
  --chat-kwargs JSON  merged into chat_template_kwargs, e.g. '{"reasoning_effort":"low"}'.

A third (L70m, docs/next-model-plan.md) replaces the server's template for requests that
carry tools, and is recorded the same way:

  --render llama31-meta-json
      Meta's documented JSON tool calling for Llama 3.1 instead of the HF template, which
      tells the model to "respond with a JSON for a function call". Rendered here as text and
      sent to /tokenize as a prompt; vLLM refuses a per-request chat_template.
      - system and tool turns: verbatim from llama-models models/llama3_1/prompt_format.md
        @ 8d29d93f, "JSON based tool calling";
      - tool list: the harness's own schemas, json indent 4, one newline apart
        (prompt_templates/system_prompts.py JsonCustomToolGenerator; its every-parameter-is-
        "object" rewrite is not copied, so Llama sees the parameter types the others see);
      - a prior call: "<|python_tag|>" + any text before it + {"type", "name", "parameters"}
        + "<|eom_id|>" (api/chat_format.py encode_message, api/tool_utils.py
        encode_tool_call);
      - tool output: an ipython turn, raw.
      Requests without tools keep the server's template: with no tools the two agree.

  python3 g7a_eval.py --selfcheck
  python3 g7a_eval.py [--think-tag] [--chat-kwargs JSON] [--render NAME] <research_eval.py args>
                      (--thinking off|on; default = on here)
"""
import json
import os
import re
import sys

sys.dont_write_bytecode = True  # no __pycache__ in ~/gpu-lab/bench
sys.path.insert(0, os.path.expanduser("~/gpu-lab/bench"))
import research_eval as rev  # noqa: E402

XML_CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
XML_PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?(?:</parameter>|(?=<parameter=)|$)", re.S)
GLM_CALL = re.compile(r"<tool_call>\s*([^<>{}\s]+)\s*(.*?)(?:</tool_call>|$)", re.S)
GLM_ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)(?:</arg_value>|$)", re.S)
_parse = rev.parse_tool_call
_split = rev.split_think
THINK_OPEN = [False]


def parse_tool_call(text):
    got = _parse(text)
    if got is not None:
        return got
    m = XML_CALL.search(text)
    if m and m.group(1) in rev.TOOL_NAMES:
        args = {k: v for k, v in XML_PARAM.findall(m.group(2))}
        return m.group(1), args, text[: m.start()].strip(), "xml_function"
    m = GLM_CALL.search(text)
    if m and m.group(1) in rev.TOOL_NAMES:
        args = {k.strip(): v for k, v in GLM_ARG.findall(m.group(2))}
        return m.group(1), args, text[: m.start()].strip(), "glm_arg"
    return None


def split_think(text):
    if THINK_OPEN[0] and "</think>" not in text:
        return "", True
    return _split(text)


LLAMA31_SYSTEM = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython\n\n"
                  "Cutting Knowledge Date: December 2023\nToday Date: 21 September 2024\n\n"
                  "You are a helpful assistant.\n<|eot_id|>")
LLAMA31_TOOLS = ("Answer the user's question by making use of the following functions if needed.\n"
                 "If none of the function can be used, please say so.\n"
                 "Here is a list of functions in JSON format:\n")


STR_LIST = re.compile(r'\[\n\s+("[^"\n]*"(?:,\n\s+"[^"\n]*")*)\n\s+\]')


def llama31_tool(t):
    """One tool as Meta's generator lays it out: indent 4, a list of names ("required") on one
    line, as its tojson writes it."""
    return STR_LIST.sub(lambda m: "[" + re.sub(r",\n\s+", ", ", m.group(1)) + "]",
                        json.dumps(t, indent=4, ensure_ascii=False))


def llama31_meta_json(messages, tools):
    """The prompt text for a chat with tools in Meta's documented Llama 3.1 JSON format."""
    def turn(role, text, end="<|eot_id|>"):
        return f"<|start_header_id|>{role}<|end_header_id|>\n\n{text}{end}"
    listing = "\n".join(llama31_tool(t) for t in tools)
    out = [LLAMA31_SYSTEM, turn("user", LLAMA31_TOOLS + listing + "\n\nReturn function calls in JSON format.")]
    for m in messages:
        if m["role"] == "user":
            out.append(turn("user", m["content"]))
        elif m["role"] == "assistant" and m.get("tool_calls"):
            calls = "".join(json.dumps({"type": "function", "name": c["function"]["name"],
                                        "parameters": json.loads(c["function"]["arguments"])})
                            for c in m["tool_calls"])
            out.append(turn("assistant", "<|python_tag|>" + (m.get("content") or "") + calls, "<|eom_id|>"))
        elif m["role"] == "tool":
            out.append(turn("ipython", m["content"]))
        else:  # the harness's history holds only these three
            raise ValueError(f"llama31-meta-json cannot render a {m['role']} message")
    return "".join(out) + "<|start_header_id|>assistant<|end_header_id|>\n\n"


RENDERS = {"llama31-meta-json": llama31_meta_json}
_post = rev.post
WRAP = {"think_tag": None, "chat_kwargs": {}, "render": None}


def shape(body):
    """The /tokenize request with this wrapper's options applied (a copy)."""
    body = dict(body)
    if WRAP["think_tag"]:
        body["messages"] = [{"role": "system", "content": WRAP["think_tag"]}] + body["messages"]
    if WRAP["chat_kwargs"]:
        body["chat_template_kwargs"] = {**body.get("chat_template_kwargs", {}), **WRAP["chat_kwargs"]}
    if WRAP["render"] and body.get("tools"):
        body = {"model": body["model"], "prompt": RENDERS[WRAP["render"]](body["messages"], body["tools"]),
                "add_special_tokens": False}
    return body


def post(base, path, body, *args, **kw):
    return _post(base, path, shape(body) if path == "/tokenize" else body, *args, **kw)


def selfcheck():
    ok = True

    def check(name, got, want):
        nonlocal ok
        ok &= got == want
        print(("ok  " if got == want else "FAIL"), name, got if got != want else "")

    call = "I'll check.\n<tool_call>\n<function=bash>\n<parameter=command>\njq --help\n</parameter>\n</function>\n</tool_call>"
    check("xml call", parse_tool_call(call), ("bash", {"command": "jq --help"}, "I'll check.", "xml_function"))
    check("xml multiline value", parse_tool_call("<tool_call>\n<function=bash>\n<parameter=command>\nman xz\nx\n</parameter>\n</function>\n</tool_call>"),
          ("bash", {"command": "man xz\nx"}, "", "xml_function"))
    check("xml unclosed (stop token)", parse_tool_call("<tool_call>\n<function=web_search>\n<parameter=query>\nrsync flags\n"),
          ("web_search", {"query": "rsync flags"}, "", "xml_function"))
    check("xml unknown tool", parse_tool_call("<tool_call>\n<function=python>\n<parameter=code>\n1\n</parameter>\n</function>\n</tool_call>"), None)
    check("xml no call", parse_tool_call("The flag is --foo."), None)
    check("glm call", parse_tool_call("I'll check.\n<tool_call>bash<arg_key>command</arg_key><arg_value>jq --help"
                                      "</arg_value></tool_call>"), ("bash", {"command": "jq --help"}, "I'll check.", "glm_arg"))
    check("glm newlines, multiline value", parse_tool_call("<tool_call>bash\n<arg_key>command</arg_key>\n"
                                                         "<arg_value>man xz\nx</arg_value>\n</tool_call>"),
          ("bash", {"command": "man xz\nx"}, "", "glm_arg"))
    check("glm unclosed (stop token)", parse_tool_call("<tool_call>web_search<arg_key>query</arg_key><arg_value>rsync flags"),
          ("web_search", {"query": "rsync flags"}, "", "glm_arg"))
    check("glm unknown tool", parse_tool_call("<tool_call>python<arg_key>code</arg_key><arg_value>1</arg_value></tool_call>"), None)
    check("glm does not take qwen json", parse_tool_call('<tool_call>{"name": "python", "arguments": {}}</tool_call>'), None)
    qwen = '<tool_call>\n{"name": "bash", "arguments": {"command": "jq --help"}}\n</tool_call>'
    check("qwen json unchanged", parse_tool_call(qwen), _parse(qwen))
    llama = '{"name": "bash", "parameters": {"command": "jq --help"}}'
    check("llama json unchanged", parse_tool_call(llama), _parse(llama))
    THINK_OPEN[0] = True
    check("think open, closed", split_think("reasoning\n</think>\n\nAnswer."), ("Answer.", False))
    check("think open, truncated", split_think("reasoning that never ends"), ("", True))
    THINK_OPEN[0] = False
    check("think off, plain", split_think("Answer."), ("Answer.", False))
    check("think off, same as harness", split_think("<think>x"), _split("<think>x"))
    nano9b = '<TOOLCALL>[{"name": "bash", "arguments": {"command": "jq --help"}}]</TOOLCALL>'
    check("nano 9b toolcall", parse_tool_call(nano9b)[:2], ("bash", {"command": "jq --help"}))
    req = {"messages": [{"role": "user", "content": "q"}], "chat_template_kwargs": {"enable_thinking": False}}
    check("no options, request unchanged", shape(req), req)
    WRAP.update(think_tag="/no_think", chat_kwargs={"reasoning_effort": "low"})
    check("options applied", shape(req), {"messages": [{"role": "system", "content": "/no_think"},
                                                       {"role": "user", "content": "q"}],
                                          "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}})
    check("request not mutated", req["messages"], [{"role": "user", "content": "q"}])
    WRAP.update(think_tag=None, chat_kwargs={})

    # llama31-meta-json: the expected text is typed out from Meta's doc, not built by the code.
    tools = [{"type": "function", "function": {"name": "bash", "description": "d"}},
             {"type": "function", "function": {"name": "web_search", "description": "w"}}]
    first = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython\n\n"
             "Cutting Knowledge Date: December 2023\nToday Date: 21 September 2024\n\nYou are a helpful assistant.\n"
             "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
             "Answer the user's question by making use of the following functions if needed.\n"
             "If none of the function can be used, please say so.\nHere is a list of functions in JSON format:\n"
             '{\n    "type": "function",\n    "function": {\n        "name": "bash",\n        "description": "d"\n    }\n}\n'
             '{\n    "type": "function",\n    "function": {\n        "name": "web_search",\n        "description": "w"\n    }\n}'
             "\n\nReturn function calls in JSON format.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
             "q<|eot_id|>")
    gen = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    WRAP.update(render="llama31-meta-json")
    treq = {"model": "m", "messages": [{"role": "user", "content": "q"}], "add_generation_prompt": True,
            "tools": tools, "chat_template_kwargs": {"enable_thinking": False}}
    check("llama31 first turn", shape(treq), {"model": "m", "prompt": first + gen, "add_special_tokens": False})
    hist = treq["messages"] + [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function", "function": {
            "name": "bash", "arguments": json.dumps({"command": "jq --help"})}}]},
        {"role": "tool", "tool_call_id": "call_0", "name": "bash", "content": "out1"},
        {"role": "assistant", "content": "Checking.", "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "web_search", "arguments": json.dumps({"query": "x"})}}]},
        {"role": "tool", "tool_call_id": "call_1", "name": "web_search", "content": "out2"}]
    two = (first + "<|start_header_id|>assistant<|end_header_id|>\n\n<|python_tag|>"
           '{"type": "function", "name": "bash", "parameters": {"command": "jq --help"}}<|eom_id|>'
           "<|start_header_id|>ipython<|end_header_id|>\n\nout1<|eot_id|>"
           "<|start_header_id|>assistant<|end_header_id|>\n\n<|python_tag|>Checking."
           '{"type": "function", "name": "web_search", "parameters": {"query": "x"}}<|eom_id|>'
           "<|start_header_id|>ipython<|end_header_id|>\n\nout2<|eot_id|>" + gen)
    check("llama31 two-call history", shape(dict(treq, messages=hist))["prompt"], two)
    check("llama31 required on one line", llama31_tool({"type": "function", "function": {"name": "b", "parameters": {
        "type": "object", "properties": {"c": {"type": "string"}}, "required": ["c", "d"]}}}),
        '{\n    "type": "function",\n    "function": {\n        "name": "b",\n        "parameters": {\n'
        '            "type": "object",\n            "properties": {\n                "c": {\n'
        '                    "type": "string"\n                }\n            },\n            "required": ["c", "d"]\n'
        '        }\n    }\n}')
    check("llama31 no tools, request unchanged", shape(req), req)
    meta = '{\n    "type": "function",\n    "name": "bash",\n    "parameters": {\n        "command": "jq --help"\n    }\n}'
    check("llama31 Meta's response form parses", parse_tool_call(meta), ("bash", {"command": "jq --help"}, "", "json"))
    WRAP.update(render=None)
    print("selfcheck", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--selfcheck"]:
        sys.exit(selfcheck())
    # Lightning's template defaults enable_thinking to True, so the harness's "default"
    # leaves the prompt ending in an open <think> exactly like "on".
    argv = sys.argv[1:]
    if "--think-tag" in argv:
        argv.remove("--think-tag")
        WRAP["think_tag"] = True
    if "--chat-kwargs" in argv:
        i = argv.index("--chat-kwargs")
        WRAP["chat_kwargs"] = json.loads(argv[i + 1])
        del argv[i:i + 2]
    if "--render" in argv:
        i = argv.index("--render")
        WRAP["render"] = argv[i + 1]
        del argv[i:i + 2]
        if WRAP["render"] not in RENDERS:
            sys.exit(f"--render: one of {', '.join(RENDERS)}")
        if WRAP["think_tag"]:
            sys.exit("--render replaces the template's system turn, so not with --think-tag")
    thinking = argv[argv.index("--thinking") + 1] if "--thinking" in argv else "default"
    THINK_OPEN[0] = thinking in ("on", "default")
    if WRAP["think_tag"]:
        if thinking == "default":
            sys.exit("--think-tag needs --thinking on|off")
        WRAP["think_tag"] = "/think" if thinking == "on" else "/no_think"
    rev.parse_tool_call = parse_tool_call
    rev.split_think = split_think
    rev.post = post
    sys.argv = [rev.__file__] + argv
    rev.main()
    if (WRAP["think_tag"] or WRAP["chat_kwargs"] or WRAP["render"]) and "--out" in argv:
        out = argv[argv.index("--out") + 1]
        res = json.load(open(out))
        res["wrapper"] = WRAP
        json.dump(res, open(out, "w"), indent=1)

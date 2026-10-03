"""G7a (plan.md): ~/gpu-lab/bench/research_eval.py, unchanged, on Nemotron 3.5 Lightning.
Two patches, both in the harness's reading of the model's text; the scorer is untouched:

  parse_tool_call  also reads the Qwen3-Coder XML form Lightning's template asks for
                   (<tool_call><function=NAME><parameter=K>V</parameter></function>), after
                   the harness's own Qwen-JSON and Llama forms have failed to match.
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

  python3 g7a_eval.py --selfcheck
  python3 g7a_eval.py [--think-tag] [--chat-kwargs JSON] <research_eval.py args>
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
    return None


def split_think(text):
    if THINK_OPEN[0] and "</think>" not in text:
        return "", True
    return _split(text)


_post = rev.post
WRAP = {"think_tag": None, "chat_kwargs": {}}


def shape(body):
    """The /tokenize request with this wrapper's options applied (a copy)."""
    body = dict(body)
    if WRAP["think_tag"]:
        body["messages"] = [{"role": "system", "content": WRAP["think_tag"]}] + body["messages"]
    if WRAP["chat_kwargs"]:
        body["chat_template_kwargs"] = {**body.get("chat_template_kwargs", {}), **WRAP["chat_kwargs"]}
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
    if (WRAP["think_tag"] or WRAP["chat_kwargs"]) and "--out" in argv:
        out = argv[argv.index("--out") + 1]
        res = json.load(open(out))
        res["wrapper"] = WRAP
        json.dump(res, open(out, "w"), indent=1)

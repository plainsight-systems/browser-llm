#!/usr/bin/env python3
"""Builds the chat-template fixtures for tests/web/template.test.js.

For each model in web/models.json it reads the chat template and the BOS and
EOS token text from the real model file's header (an HTTP range fetch, not a
download), then renders a fixed set of conversations with Python's jinja2,
configured the way Hugging Face transformers configures it. Those renderings
are the reference the page's template engine is checked against: an
independent implementation of the same language.

Needs network access and jinja2 (pip install jinja2). Not run in CI; its output
is committed.

    python3 tools/make_template_fixtures.py tests/web/fixtures/templates.json
"""
import json
import struct
import sys
import urllib.request

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

HEADER_BYTES = 16 * 2**20
FIXED_DATE = "29 Sep 2026"

# gguf_type
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
SCALAR = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
          F32: "<f", BOOL: "<?", U64: "<Q", I64: "<q", F64: "<d"}


def fetch_header(url):
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{HEADER_BYTES - 1}"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def read_metadata(data):
    """Returns {key: value}; string arrays are decoded, others skipped."""
    pos = 0

    def take(fmt):
        nonlocal pos
        (value,) = struct.unpack_from(fmt, data, pos)
        pos += struct.calcsize(fmt)
        return value

    def take_string():
        nonlocal pos
        length = take("<Q")
        text = data[pos:pos + length].decode("utf-8", errors="replace")
        pos += length
        return text

    def take_value(vtype):
        if vtype == STRING:
            return take_string()
        if vtype == ARRAY:
            etype, count = take("<I"), take("<Q")
            return [take_value(etype) for _ in range(count)]
        return take(SCALAR[vtype])

    assert data[:4] == b"GGUF"
    pos = 8
    take("<Q")  # tensor count
    metadata = {}
    for _ in range(take("<Q")):
        key = take_string()
        metadata[key] = take_value(take("<I"))
    return metadata


def environment():
    # As transformers builds it (chat_template_utils._compile_jinja_template).
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: FIXED_DATE if fmt == "%d %b %Y" else fmt
    return env


CONVERSATIONS = [
    {"name": "one user turn",
     "messages": [{"role": "user", "content": "Hello"}]},
    {"name": "system prompt",
     "messages": [{"role": "system", "content": "Be brief."},
                  {"role": "user", "content": "What is a KV cache?"}]},
    {"name": "second turn",
     "messages": [{"role": "user", "content": "Hi"},
                  {"role": "assistant", "content": "Hello! How can I help?"},
                  {"role": "user", "content": "Tell me a joke."}]},
    {"name": "earlier reasoning in the history",
     "messages": [{"role": "user", "content": "2+2?"},
                  {"role": "assistant", "content": "<think>\nAdd them.\n</think>\n\n4"},
                  {"role": "user", "content": "And 3+3?"}]},
]


def main():
    out = sys.argv[1]
    models = json.load(open("web/models.json"))["models"]
    env = environment()
    fixtures = []
    for model in models:
        metadata = read_metadata(fetch_header(model["url"]))
        tokens = metadata["tokenizer.ggml.tokens"]
        # A model may declare no BOS token (Qwen3 does not).
        token = lambda key: tokens[metadata[key]] if key in metadata else None
        chat = {
            "template": metadata["tokenizer.chat_template"],
            "bosToken": token("tokenizer.ggml.bos_token_id"),
            "eosToken": token("tokenizer.ggml.eos_token_id"),
        }
        template = env.from_string(chat["template"])
        thinking = [None, False] if "enable_thinking" in chat["template"] else [None]
        cases = []
        for conversation in CONVERSATIONS:
            for enable_thinking in thinking:
                variables = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
                expected = template.render(messages=conversation["messages"],
                                           add_generation_prompt=True,
                                           bos_token=chat["bosToken"],
                                           eos_token=chat["eosToken"], **variables)
                cases.append({"name": conversation["name"], "messages": conversation["messages"],
                              "variables": variables, "expected": expected})
        fixtures.append({"model": model["id"], "chat": chat, "cases": cases})
        print(f"{model['id']}: {len(cases)} cases", file=sys.stderr)
    with open(out, "w") as f:
        json.dump({"date": FIXED_DATE, "fixtures": fixtures}, f, indent=1, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()

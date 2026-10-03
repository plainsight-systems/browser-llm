#!/usr/bin/env python3
"""Builds the chat-template fixtures for tests/web/template.test.js.

For each model in web/models.json it reads the chat template and the BOS and
EOS token text from the model file's header, as pinned in
.cache/test-data/tokenizer/ (make test-data), then renders a fixed set of
conversations with Python's jinja2, configured the way Hugging Face
transformers configures it. Those renderings are the reference the page's
template engine is checked against: an independent implementation of the
same language.

Neither the templates nor their renderings are committed: they are the
model publishers' text, under the models' licenses. The fixture holds the
conversations and the SHA-256 of each reference rendering; the test reads
the template from the same pinned header and compares hashes. To see a
reference rendering, run with --print.

Needs jinja2 (pip install -r tools/requirements.txt). Not run in CI; its
output is committed.

    python3 tools/make_template_fixtures.py tests/web/fixtures/templates.json
    python3 tools/make_template_fixtures.py --print [model-id]
"""
import hashlib
import json
import sys

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

from gguf_header import read_metadata

HEADERS = "tests/fixtures/tokenizer/manifest.json"
CACHE = ".cache/test-data/tokenizer"

FIXED_DATE = "29 Sep 2026"


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


def renderings(model_id):
    """Yields (conversation name, variables, reference rendering) for one model
    in web/models.json, from its header as pinned in .cache/test-data/."""
    pinned = json.load(open(HEADERS))
    if model_id not in pinned:
        sys.exit(f"{model_id}: no header pinned in {HEADERS}")
    with open(f"{CACHE}/{model_id}.header", "rb") as f:
        header = f.read()
    if hashlib.sha256(header).hexdigest() != pinned[model_id]["header"]["sha256"]:
        sys.exit(f"{model_id}: the cached header is not the pinned one; run make test-data")
    metadata, _ = read_metadata(header)
    tokens = metadata["tokenizer.ggml.tokens"]
    # A model may declare no BOS token (Qwen3 does not).
    token = lambda key: tokens[metadata[key]] if key in metadata else None
    bos, eos = token("tokenizer.ggml.bos_token_id"), token("tokenizer.ggml.eos_token_id")
    source = metadata["tokenizer.chat_template"]
    template = environment().from_string(source)
    thinking = [None, False] if "enable_thinking" in source else [None]
    for conversation in CONVERSATIONS:
        for enable_thinking in thinking:
            variables = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
            yield conversation, variables, template.render(
                messages=conversation["messages"], add_generation_prompt=True,
                bos_token=bos, eos_token=eos, **variables)


def main():
    printing = sys.argv[1:2] == ["--print"]
    only = sys.argv[2] if printing and len(sys.argv) > 2 else None
    fixtures = []
    for model in json.load(open("web/models.json"))["models"]:
        cases = []
        for conversation, variables, rendered in renderings(model["id"]):
            if printing and only in (None, model["id"]):
                print(f"=== {model['id']}: {conversation['name']} {json.dumps(variables)}\n{rendered}")
            cases.append({"name": conversation["name"], "messages": conversation["messages"],
                          "variables": variables,
                          "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest()})
        fixtures.append({"model": model["id"], "cases": cases})
        print(f"{model['id']}: {len(cases)} cases", file=sys.stderr)
    if printing:
        return
    with open(sys.argv[1], "w") as f:
        json.dump({"date": FIXED_DATE, "fixtures": fixtures}, f, indent=1, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Builds the tokenizer fixtures: for each model, texts and the token IDs they
must encode to.

Two independent references encode every text, and a case is written only when
they agree:

  - Hugging Face tokenizers, on the model's own tokenizer.json from the
    publisher's repository. This is the tokenizer the model was trained with.
  - llama.cpp's llama-tokenize, on the GGUF header itself. This reads the same
    vocabulary arrays our tokenizer will.

Before encoding anything it checks that tokenizer.json and the GGUF hold the
same tokenizer: every token string, and for BPE every merge in order. A
mismatch stops the run.

Where the references disagree, the run stops and prints the cases. A
disagreement is settled by a decision recorded in DECISIONS below, naming the
reference this harness follows and why; nothing is settled silently.

The encoding never adds BOS: the chat template writes it as text, and special
tokens written as text encode as their single IDs.

For a byte-level BPE model it also records how its pre-tokenizer splits every
text, from tokenizer.json's own split pattern run alone, so a wrong split is
caught as a wrong piece before any merging. The pre-tokenizer is named as the
GGUF names it, and one name must always mean one pattern.

Output, under the given directory:
  <model>.inc        the cases, as C++ initializers the tokenizer tests include
  split-<pre>.inc    each text and the pieces the pre-tokenizer splits it into
  manifest.json      each model's header (URL, bytes, SHA-256) and the references

Needs network access, a Hugging Face login with the Llama 3.2 and Gemma 3
licences accepted, the packages in tools/requirements.txt, and a build of
llama.cpp's llama-tokenize. Not run in CI; its output is committed.

    python3 tools/make_tokenizer_fixtures.py <llama-tokenize> <llama.cpp commit> tests/fixtures/tokenizer
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unicodedata

import tokenizers
from huggingface_hub import HfApi, hf_hub_download

from gguf_header import fetch_header, read_metadata, without_tensors

# The publisher's repository for each model's tokenizer.json.
REFERENCE_REPOS = {
    "qwen3-0.6b-q4_0": "Qwen/Qwen3-0.6B",
    "llama-3.2-1b-instruct-q4_0": "meta-llama/Llama-3.2-1B-Instruct",
    "gemma-3-1b-it-q4_0": "google/gemma-3-1b-it",
}

# token_type values in a GGUF vocabulary.
USER_DEFINED, UNUSED = 4, 5

# Each text is chosen for what it can catch. Every model encodes every text.
TEXTS = [
    ("empty", ""),
    ("one word", "Hello"),
    ("sentence", "The quick brown fox jumps over the lazy dog."),
    ("leading space", " leading"),
    ("trailing space", "trailing "),
    ("space runs", "a  b   c    d"),
    ("tabs", "\tindented\t\ttwice"),
    ("newline runs", "one\n\ntwo\n\n\nthree"),
    ("windows newlines", "one\r\ntwo\r\n"),
    ("spaces before a newline", "end  \nnext"),
    # Qwen's split takes digits one at a time; Llama's up to three.
    ("digits", "12345678"),
    ("numbers in prose", "In 2026 there were 1,234,567 tokens and 3.14159 pies."),
    # Both splits match contractions case-insensitively.
    ("contractions", "don't won't I'M they'll we'd you've she's"),
    ("punctuation runs", "Wait... what?!?! (really) [yes] {no} <maybe>"),
    ("code", "def f(x):\n    return x**2 + 1  # square\n"),
    ("url", "https://example.com/a/b?c=1&d=two#frag"),
    ("backslashes", "C:\\path\\to\\file and \\n is not a newline"),
    # The same words composed and decomposed: a normalizer treats them alike.
    ("accents, precomposed", "café naïve résumé"),
    ("accents, decomposed", "cafe\u0301 nai\u0308ve re\u0301sume\u0301"),
    ("chinese and japanese", "我喜欢学习新的语言。日本語も少し。"),
    ("korean", "안녕하세요"),
    ("arabic", "مرحبا بالعالم"),
    ("devanagari", "नमस्ते दुनिया"),
    ("mixed scripts", "Tokyo 東京 2026 🚄"),
    ("emoji", "I 💙 tokens 🎉"),
    ("emoji joined by zero-width joiners", "family: 👨\u200d👩\u200d👧\u200d👦"),
    ("flags", "🇳🇿🇯🇵"),
    ("outside the basic multilingual plane", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𐍈"),
    ("zero-width space", "zero\u200bwidth"),
    # Characters a vocabulary may lack, which Gemma spells out as bytes.
    ("control characters", "a\x01b\x7fc"),
    # Special-token text typed into a message encodes as the special token.
    ("special-token text typed by a user", "<|im_end|><start_of_turn><|eot_id|>"),
    ("repetition", "a" * 50 + " " + "ab" * 20),
    ("paragraph",
     "Tokenizers turn text into numbers. A model never sees letters, only the "
     "identifiers of pieces it learned while training, and the same sentence can "
     "split differently in two models that share an algorithm."),
]

# Disagreements between the references, settled: (model, case name) ->
# ("hf" or "llama.cpp", why this harness follows that one).
DECISIONS = {
    ("qwen3-0.6b-q4_0", "accents, decomposed"): (
        "hf", "Qwen's tokenizer.json normalizes text to NFC before splitting it, as "
              "the model saw text in training; llama.cpp does not normalize. Text "
              "arrives decomposed in practice: macOS stores file names that way."),
}


def tokenizer_json(repo):
    """The repository's current tokenizer.json, pinned to its revision."""
    revision = HfApi().model_info(repo).sha
    path = hf_hub_download(repo, "tokenizer.json", revision=revision)
    return path, revision


def check_same_tokenizer(model_id, metadata, path):
    """Stops the run unless tokenizer.json and the GGUF hold one tokenizer."""
    reference = json.loads(pathlib.Path(path).read_text())
    by_id = {i: token for token, i in reference["model"]["vocab"].items()}
    by_id.update({t["id"]: t["content"] for t in reference["added_tokens"]})
    tokens = metadata["tokenizer.ggml.tokens"]
    types = metadata["tokenizer.ggml.token_type"]
    sentencepiece = metadata["tokenizer.ggml.model"] == "llama"
    problems = []
    for i, token in enumerate(tokens):
        if i not in by_id:
            # Padding the GGUF adds past the reference vocabulary.
            if types[i] != UNUSED:
                problems.append(f"token {i} {token!r} is missing from tokenizer.json")
            continue
        expected = by_id[i]
        if sentencepiece and types[i] == USER_DEFINED:
            # llama.cpp's converter writes a SentencePiece user-defined token's
            # "▁" as the space it stands for: such tokens are matched in the
            # raw text, before spaces become "▁".
            expected = expected.replace("▁", " ")
        if token != expected:
            problems.append(f"token {i}: GGUF {token!r}, tokenizer.json {by_id[i]!r}")
    # Tokens past the GGUF's vocabulary, such as Gemma 3's image token, are
    # dropped by the converter: the model has no embedding row for them.
    beyond = sorted(i for i in by_id if i >= len(tokens))
    if beyond:
        print(f"{model_id}: tokenizer.json tokens past the vocabulary, not in the GGUF: "
              + ", ".join(f"{i} {by_id[i]!r}" for i in beyond), file=sys.stderr)
    if "tokenizer.ggml.merges" in metadata:
        merges = [m if isinstance(m, str) else " ".join(m) for m in reference["model"]["merges"]]
        if merges != metadata["tokenizer.ggml.merges"]:
            problems.append("the merges differ")
    if problems:
        sys.exit(f"{model_id}: tokenizer.json and the GGUF differ:\n  " + "\n  ".join(problems[:20]))


def llama_cpp_ids(llama_tokenize, vocabulary, text):
    with tempfile.NamedTemporaryFile(suffix=".txt") as prompt:
        prompt.write(text.encode("utf-8"))
        prompt.flush()
        # --no-escape: otherwise a typed "\n" becomes a newline.
        result = subprocess.run(
            [llama_tokenize, "-m", vocabulary, "-f", prompt.name, "--ids", "--no-bos",
             "--no-escape", "--log-disable"],
            capture_output=True, check=True)
    return json.loads(result.stdout)


def rendered_prompts(model_id):
    """The chat-template fixtures' renderings: real prompts, special tokens and all."""
    fixtures = json.loads(pathlib.Path("tests/web/fixtures/templates.json").read_text())["fixtures"]
    (model,) = [f for f in fixtures if f["model"] == model_id]
    return [(f"prompt: {c['name']} {json.dumps(c['variables'])}", c["expected"]) for c in model["cases"]]


def cpp_string(text):
    """A C++ string literal holding `text` as UTF-8. Visible characters are
    written as themselves; anything invisible, as octal escapes of its bytes
    (octal stops after three digits, where a hex escape would run on)."""
    out = []
    for ch in text:
        if ch in '"\\':
            out.append("\\" + ch)
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") or (ch != " " and ch.isspace()):
            out.append("".join(f"\\{b:03o}" for b in ch.encode("utf-8")))
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def split_pattern(model_id, path):
    """The split pattern of a byte-level tokenizer.json. Stops the run unless
    its pre-tokenizer is that one split, isolating each match, and then the
    byte mapping alone: otherwise a split run by itself is not the tokenizer's."""
    pre = json.loads(pathlib.Path(path).read_text())["pre_tokenizer"]
    steps = pre["pretokenizers"] if pre["type"] == "Sequence" else [pre]
    if (len(steps) != 2 or steps[0]["type"] != "Split" or "Regex" not in steps[0]["pattern"]
            or steps[0]["behavior"] != "Isolated" or steps[0]["invert"]
            or steps[1]["type"] != "ByteLevel" or steps[1]["use_regex"]):
        sys.exit(f"{model_id}: a pre-tokenizer other than one isolating split and the byte mapping")
    return steps[0]["pattern"]["Regex"]


def write_splits(path, pre, pattern, repo, revision, cases):
    lines = [
        f"// Generated by tools/make_tokenizer_fixtures.py for the {pre} pre-tokenizer. Do not edit.",
        f"// Hugging Face tokenizers {tokenizers.__version__}, splitting with the pattern in",
        f"// {repo} at {revision}:",
        f"// {pattern}",
        "// Each case: name, text, the pieces it splits into.",
    ]
    for name, text, pieces in cases:
        lines.append(f"{{{cpp_string(name)}, {cpp_string(text)}, {{{', '.join(map(cpp_string, pieces))}}}}},")
    path.write_text("\n".join(lines) + "\n")


def write_cases(path, model_id, repo, revision, llama_commit, cases):
    lines = [
        f"// Generated by tools/make_tokenizer_fixtures.py for {model_id}. Do not edit.",
        f"// Hugging Face tokenizers {tokenizers.__version__} on {repo} at {revision},",
        f"// agreeing with llama.cpp {llama_commit} on the GGUF header.",
        "// Each case: name, text, the token IDs it encodes to.",
    ]
    for name, text, ids in cases:
        lines.append(f"{{{cpp_string(name)}, {cpp_string(text)}, {{{', '.join(map(str, ids))}}}}},")
    path.write_text("\n".join(lines) + "\n")


def main():
    if len(sys.argv) != 4:
        sys.exit(f"usage: {sys.argv[0]} <llama-tokenize> <llama.cpp commit> <out dir>")
    llama_tokenize, llama_commit, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
    models = json.loads(pathlib.Path("web/models.json").read_text())["models"]

    disagreements, written, manifest, splits = [], {}, {}, {}
    with tempfile.TemporaryDirectory() as scratch:
        for model in models:
            model_id = model["id"]
            header = fetch_header(model["url"])
            metadata, end = read_metadata(header)
            vocabulary = pathlib.Path(scratch, f"{model_id}.gguf")
            vocabulary.write_bytes(without_tensors(header, end))

            repo = REFERENCE_REPOS[model_id]
            path, revision = tokenizer_json(repo)
            check_same_tokenizer(model_id, metadata, path)
            reference = tokenizers.Tokenizer.from_file(path)

            cases, settled = [], 0
            for name, text in TEXTS + rendered_prompts(model_id):
                hf = reference.encode(text, add_special_tokens=False).ids
                cpp = llama_cpp_ids(llama_tokenize, str(vocabulary), text)
                if hf == cpp:
                    cases.append((name, text, hf))
                elif (model_id, name) in DECISIONS:
                    follows, _why = DECISIONS[(model_id, name)]
                    cases.append((name, text, hf if follows == "hf" else cpp))
                    settled += 1
                else:
                    disagreements.append((model_id, name, text, hf, cpp))

            if metadata["tokenizer.ggml.model"] == "gpt2":
                pre = metadata["tokenizer.ggml.pre"]
                pattern = split_pattern(model_id, path)
                if pre in splits and splits[pre][0] != pattern:
                    sys.exit(f"{model_id}: pre-tokenizer {pre} has a different pattern from another model's")
                splitter = tokenizers.pre_tokenizers.Split(tokenizers.Regex(pattern), behavior="isolated")
                split_cases = []
                for name, text in TEXTS + rendered_prompts(model_id):
                    pieces = [piece for piece, _ in splitter.pre_tokenize_str(text)]
                    assert "".join(pieces) == text, f"{pre} / {name}: pieces do not cover the text"
                    split_cases.append((name, text, pieces))
                splits[pre] = (pattern, repo, revision, split_cases)

            written[model_id] = (repo, revision, cases)
            manifest[model_id] = {
                "header": {"url": model["url"], "bytes": len(header),
                           "sha256": hashlib.sha256(header).hexdigest()},
                "reference": {"repo": repo, "revision": revision,
                              "tokenizers": tokenizers.__version__, "llama.cpp": llama_commit},
            }
            print(f"{model_id}: {len(cases)} cases, {len(cases) - settled} where the references "
                  f"agree and {settled} settled in DECISIONS", file=sys.stderr)

    if disagreements:
        for model_id, name, text, hf, cpp in disagreements:
            print(f"\n{model_id} / {name}\n  text  {text!r}\n  hf    {hf}\n  cpp   {cpp}")
        sys.exit(f"\n{len(disagreements)} disagreements; settle each in DECISIONS. Nothing written.")

    out.mkdir(parents=True, exist_ok=True)
    for model_id, (repo, revision, cases) in written.items():
        write_cases(out / f"{model_id}.inc", model_id, repo, revision, llama_commit, cases)
    for pre, (pattern, repo, revision, cases) in splits.items():
        write_splits(out / f"split-{pre}.inc", pre, pattern, repo, revision, cases)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()

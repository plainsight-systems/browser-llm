#!/usr/bin/env python3
"""Fetches the test data listed in tests/fixtures/external.json, and the model
headers pinned in tests/fixtures/tokenizer/manifest.json.

Some test inputs are too large to commit, or are licensed for use rather than
redistribution: Unicode's normalization conformance file, model headers. Each
is listed with its URL and SHA-256, downloaded once into .cache/test-data/,
and verified. A file already present with the right hash is not fetched
again. A hash that does not match fails the run and keeps nothing: a test
must never read data other than what was pinned.

Standard library only, so CI runs it without installing anything.

    python3 tools/fetch_test_data.py
"""
import hashlib
import json
import pathlib
import sys
import urllib.request

MANIFEST = pathlib.Path("tests/fixtures/external.json")
# Written by tools/make_tokenizer_fixtures.py, which pins each header it read.
TOKENIZER_MANIFEST = pathlib.Path("tests/fixtures/tokenizer/manifest.json")
DESTINATION = pathlib.Path(".cache/test-data")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def download(url, length):
    headers = {"Range": f"bytes=0-{length - 1}"} if length else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
        return response.read()


def entries():
    files = json.loads(MANIFEST.read_text())["files"]
    for model_id, pins in json.loads(TOKENIZER_MANIFEST.read_text()).items():
        header = pins["header"]
        files.append({"name": f"tokenizer/{model_id}.header", "url": header["url"],
                      "bytes": header["bytes"], "sha256": header["sha256"]})
    return files


def main():
    files = entries()
    failed = False
    for entry in files:
        path = DESTINATION / entry["name"]
        if path.is_file() and sha256(path.read_bytes()) == entry["sha256"]:
            continue
        data = download(entry["url"], entry.get("bytes"))
        if sha256(data) != entry["sha256"]:
            print(f"{entry['name']}: SHA-256 {sha256(data)} does not match the pinned "
                  f"{entry['sha256']}; not kept", file=sys.stderr)
            failed = True
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        partial.write_bytes(data)
        partial.replace(path)
        print(f"{entry['name']}: fetched {len(data):,} bytes", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

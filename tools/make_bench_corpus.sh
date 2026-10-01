#!/bin/sh
# Writes the tokenizer benchmark's corpus: this repository's docs and C++
# sources as they stood at one commit, so every run measures the same bytes.
#
# Real English prose and real code, already in the repository: nothing to
# download. The generated Unicode tables are left out; they are hex, not text.
# The order is fixed (docs, the README, then sources, each sorted bytewise),
# and the corpus's SHA-256 is printed, so a figure quoted from a run names the
# exact text it was measured on.
#
#     tools/make_bench_corpus.sh <commit> <out>
set -eu

commit=$1
out=$2

files=$(
    tree=$(git ls-tree -r --name-only "$commit")
    printf '%s\n' "$tree" | grep -E '^docs/.*\.md$' | LC_ALL=C sort
    echo README.md
    printf '%s\n' "$tree" | grep -E '^src/.*\.(cpp|h)$' |
        grep -v '^src/core/tokenizer/unicode_tables\.cpp$' | LC_ALL=C sort
)

mkdir -p "$(dirname "$out")"
: > "$out"
for file in $files; do
    git show "$commit:$file" >> "$out"
done

count=$(printf '%s\n' "$files" | wc -l | tr -d ' ')
bytes=$(wc -c < "$out" | tr -d ' ')
sha=$(shasum -a 256 "$out" | cut -d' ' -f1)
echo "corpus: $count files from $commit, $bytes bytes, SHA-256 $sha"

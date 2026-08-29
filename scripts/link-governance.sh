#!/bin/sh
# Recreate local symlinks to the private governance repo.
#
# These links are gitignored. See docs/decisions/inherited.md for the canonical
# list of inherited governance and what is publicly readable.
#
# Fails loudly if the governance repo is not where it is expected.
set -eu

repo_root=$(cd "$(dirname "$0")/.." && pwd)
governance="$repo_root/../../plainsight-systems-governance"

if [ ! -d "$governance" ]; then
  echo "error: governance repo not found at $governance" >&2
  echo "clone plainsight-systems-governance there, or edit this script." >&2
  exit 1
fi

cd "$repo_root/docs/decisions"
for doc in \
  plainsight_policies \
  engineering_philosophies \
  product_memory_workflow \
  repo_creation_runbook \
  cpp_architecture_playbook \
  cpp_architecture_review \
  cpp_performance_playbook \
  cpp_performance_review
do
  target="../../../../plainsight-systems-governance/$doc.md"
  if [ ! -e "$target" ]; then
    echo "error: missing governance doc: $doc.md" >&2
    exit 1
  fi
  ln -sfn "$target" "$doc.md"
done

echo "linked 8 governance docs into docs/decisions/"

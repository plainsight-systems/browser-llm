#!/bin/sh
# Recreate local symlinks to the governance documents this repo inherits.
#
# These links are gitignored — they are an editor and agent convenience, not
# part of the repo. The canonical source is:
#   https://github.com/plainsight-systems/plainsight-systems-governance
#
# Fails loudly if a source repo or document is missing.
set -eu

repo_root=$(cd "$(dirname "$0")/.." && pwd)
public="$repo_root/../../plainsight-systems-governance"
internal="$repo_root/../../plainsight-systems-ops"

link_doc() {
  src_dir="$1"; rel="$2"; name="$3"
  if [ ! -e "$src_dir/$rel" ]; then
    echo "error: missing governance doc: $rel (looked in $src_dir)" >&2
    exit 1
  fi
  ln -sfn "$4" "$name"
}

if [ ! -d "$public" ]; then
  echo "error: governance repo not found at $public" >&2
  echo "clone plainsight-systems-governance there, or edit this script." >&2
  exit 1
fi

cd "$repo_root/docs/decisions"

for doc in \
  engineering_philosophies \
  product_memory_workflow \
  repo_creation_runbook \
  cpp_architecture_playbook \
  cpp_architecture_review \
  cpp_performance_playbook \
  cpp_performance_review
do
  link_doc "$public" "$doc.md" "$doc.md" \
    "../../../../plainsight-systems-governance/$doc.md"
done
echo "linked 7 public governance docs"

# Parent-org policy is internal and lives in the operations repo. Optional:
# absence is not an error, since the public governance stands alone.
if [ -d "$internal" ]; then
  link_doc "$internal" "governance/plainsight_policies.md" "plainsight_policies.md" \
    "../../../../plainsight-systems-ops/governance/plainsight_policies.md"
  echo "linked 1 internal policy doc"
else
  echo "note: operations repo not present; skipping internal policy doc"
fi

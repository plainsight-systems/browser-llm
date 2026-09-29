#!/bin/sh
# Proves the assembled site contains no development tooling.
#
# The fake runtime says in its own comments that it never ships. This checks
# the artifact, which is the only thing that makes that true.
set -eu
cd "$(dirname "$0")/.."

SITE=${1:-dist}
status=0

if [ ! -f "${SITE}/index.html" ]; then
    echo "FAIL: ${SITE}/ is not an assembled site" >&2
    exit 1
fi
if [ -e "${SITE}/dev" ]; then
    echo "FAIL: ${SITE}/dev/ is present; development tooling must not ship" >&2
    status=1
fi
if grep -rl "NOT A RUNTIME" "${SITE}" >/dev/null 2>&1; then
    echo "FAIL: the fake runtime's source is in ${SITE}/" >&2
    status=1
fi
[ "${status}" -eq 0 ] && echo "site: OK (no development tooling)"
exit "${status}"

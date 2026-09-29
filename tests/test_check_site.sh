#!/bin/sh
# Proves tools/check_site.sh fires: a site with development tooling in it
# fails, and one without passes.
set -eu
cd "$(dirname "$0")/.."

site="$(mktemp -d)"
trap 'rm -rf "${site}"' EXIT
touch "${site}/index.html"

./tools/check_site.sh "${site}" >/dev/null || { echo "FAIL: a clean site was rejected" >&2; exit 1; }

mkdir "${site}/dev"
if ./tools/check_site.sh "${site}" >/dev/null 2>&1; then
    echo "FAIL: a site containing dev/ was accepted" >&2; exit 1
fi
rmdir "${site}/dev"

cp web/dev/fake_runtime.js "${site}/runtime.js"
if ./tools/check_site.sh "${site}" >/dev/null 2>&1; then
    echo "FAIL: a site containing the fake runtime was accepted" >&2; exit 1
fi

echo "check_site: OK (both rules fire)"

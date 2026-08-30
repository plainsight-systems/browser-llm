#!/bin/sh
# Proves each boundary rule actually fires.
#
# The previous checker passed the tree while enforcing less than it claimed:
# it matched EMSCRIPTEN_KEEPALIVE rather than EMSCRIPTEN_, and permitted any
# one file to be Emscripten-aware rather than the allowlisted one. A checker
# nobody has seen fail is not evidence of anything.
set -eu

cd "$(dirname "$0")/.."
CHECK=./tools/check_boundaries.sh
CREATED=""

cleanup() { for f in ${CREATED}; do rm -f "$f"; done; }
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

expect_violation() {
    desc="$1"
    if "${CHECK}" >/dev/null 2>&1; then
        fail "${desc}: checker passed but should have failed"
    fi
}

# Baseline: the real tree must pass, or every assertion below is meaningless.
"${CHECK}" >/dev/null 2>&1 || fail "clean tree does not pass the checker"

# 1. Emscripten token in core — using a macro that is NOT the literal
#    KEEPALIVE the old checker looked for.
F=src/core/gpu/_boundary_probe.h; CREATED="${CREATED} ${F}"
printf '#pragma once\n#ifdef EMSCRIPTEN_HAS_THREADS\n#endif\n' > "${F}"
expect_violation "EMSCRIPTEN_ macro in core"
rm -f "${F}"

# 2. Angled outward include — the old pattern only caught the quoted form.
F=src/core/gpu/_boundary_probe2.h; CREATED="${CREATED} ${F}"
printf '#pragma once\n#include <wasm/bindings.h>\n' > "${F}"
expect_violation "angled include from core into wasm/"
rm -f "${F}"

# 3. A second Emscripten-aware file outside the allowlist. The old check
#    counted files, so this one slipped through whenever bindings.cpp was
#    not itself counted.
F=src/wasm/_boundary_probe3.cpp; CREATED="${CREATED} ${F}"
printf '#include <emscripten.h>\n' > "${F}"
expect_violation "second Emscripten-aware translation unit"
rm -f "${F}"

# 4. Near-collision: the allowlist is an exact path, not a regex. With `.`
#    treated as a wildcard, "bindingsXcpp" would be silently excluded too.
F=src/wasm/bindingsXcpp; CREATED="${CREATED} ${F}"
printf '#include <emscripten.h>\n' > "${F}"
expect_violation "near-collision filename slipping through the allowlist"
rm -f "${F}"

# Tree must be clean again afterwards.
"${CHECK}" >/dev/null 2>&1 || fail "checker still failing after probes removed"

echo "check_boundaries: OK (all four rules fire)"

#!/bin/sh
# Enforces the core/wrapper boundary that makes the core natively testable.
#
# A reviewer noticing a stray #include <emscripten.h> is not a control. This
# is, and it runs in CI. Covered by tests/test_check_boundaries.sh, which
# proves each rule actually fires.
set -eu

cd "$(dirname "$0")/.."

# The single translation unit permitted to be Emscripten-aware. An exact
# allowlist, not a count: "some one file" is not the invariant.
WRAPPER=src/wasm/bindings.cpp

# Matches the packet's full forbidden set, not a sample of it. EMSCRIPTEN_
# covers KEEPALIVE and every other macro in that family.
EMSCRIPTEN_TOKENS='emscripten\.h|emscripten/|EM_JS|EM_ASM|EMSCRIPTEN_'

status=0
fail() { echo "BOUNDARY VIOLATION: $1" >&2; status=1; }

# 1. Core must remain platform-neutral.
hits="$(grep -rlE "${EMSCRIPTEN_TOKENS}" src/core/ 2>/dev/null || true)"
if [ -n "${hits}" ]; then
    echo "${hits}" | sed 's/^/  /' >&2
    fail "src/core/ references Emscripten. Core must compile natively unchanged."
fi

# 2. Core must not depend outward on the wrapper or the page. Catches both
#    quoted and angled include forms, and any path mentioning them.
hits="$(grep -rnE '#[[:space:]]*include[[:space:]]*[<"](\.\./)*(wasm|web)/' src/core/ 2>/dev/null || true)"
if [ -n "${hits}" ]; then
    echo "${hits}" | sed 's/^/  /' >&2
    fail "src/core/ includes from src/wasm/ or web/. Dependency direction is inward only."
fi

# 3. Exactly the allowlisted file may be Emscripten-aware. A second one is a
#    decision, not a drift — and it must be that file, not merely one file.
offenders="$(grep -rlE "${EMSCRIPTEN_TOKENS}" src/ 2>/dev/null | grep -v "^${WRAPPER}$" || true)"
if [ -n "${offenders}" ]; then
    echo "${offenders}" | sed 's/^/  /' >&2
    fail "only ${WRAPPER} may be Emscripten-aware."
fi

if [ "${status}" -eq 0 ]; then
    echo "boundaries OK"
fi
exit "${status}"

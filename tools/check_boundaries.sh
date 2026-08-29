#!/bin/sh
# Enforces the core/wrapper boundary that makes the core natively testable.
#
# A reviewer noticing a stray #include <emscripten.h> is not a control. This
# is, and it runs in CI.
set -eu

cd "$(dirname "$0")/.."
status=0

fail() {
    echo "BOUNDARY VIOLATION: $1" >&2
    status=1
}

# 1. Core must remain platform-neutral.
if grep -rlE 'emscripten\.h|EM_JS|EM_ASM|EMSCRIPTEN_KEEPALIVE' src/core/ 2>/dev/null | grep -q .; then
    grep -rlE 'emscripten\.h|EM_JS|EM_ASM|EMSCRIPTEN_KEEPALIVE' src/core/ | while read -r f; do
        echo "  $f" >&2
    done
    fail "src/core/ references Emscripten. Core must compile natively unchanged."
fi

# 2. Core must not depend outward on the wrapper.
if grep -rlE '#include[[:space:]]*"(wasm|web)/' src/core/ 2>/dev/null | grep -q .; then
    fail "src/core/ includes from src/wasm/ or web/. Dependency direction is inward only."
fi

# 3. The wrapper stays a single translation unit. A second one is a decision,
#    not a drift.
count=$(grep -rlE 'emscripten\.h' src/ 2>/dev/null | grep -v '^src/core/' | wc -l | tr -d ' ')
if [ "$count" -gt 1 ]; then
    fail "$count files include emscripten.h; expected at most 1 (src/wasm/bindings.cpp)."
fi

if [ "$status" -eq 0 ]; then
    echo "boundaries OK"
fi
exit "$status"

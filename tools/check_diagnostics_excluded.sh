#!/bin/sh
# Proves diagnostic instrumentation is absent from the clean build and present
# in the diagnostic one.
#
# Documenting that a symbol "should not ship" is not the same as it not
# shipping. This checks the actual artifact.
set -eu
cd "$(dirname "$0")/.."

SYMBOL=bllm_run_readback_bench
CLEAN=build/wasm-release/browser_llm.mjs
DIAG=build/wasm-diag/browser_llm.mjs
status=0
checked=0

if [ -f "${CLEAN}" ]; then
    checked=$((checked + 1))
    if grep -q "${SYMBOL}" "${CLEAN}"; then
        echo "FAIL: ${SYMBOL} is present in the clean build (${CLEAN})" >&2
        status=1
    fi
fi

if [ -f "${DIAG}" ]; then
    checked=$((checked + 1))
    if ! grep -q "${SYMBOL}" "${DIAG}"; then
        echo "FAIL: ${SYMBOL} is absent from the diagnostic build (${DIAG})" >&2
        echo "  the gate excludes it from both, so it is unreachable everywhere" >&2
        status=1
    fi
fi

if [ "${checked}" -eq 0 ]; then
    echo "diagnostics: no wasm artifacts built; nothing to check"
    exit 0
fi
[ "${status}" -eq 0 ] && echo "diagnostics: OK (${checked} artifact(s) checked)"
exit "${status}"

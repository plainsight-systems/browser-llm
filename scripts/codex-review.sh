#!/usr/bin/env bash
# codex-review.sh — independent review of a packet and the work it covers.
#
# Usage:  ./scripts/codex-review.sh <packet-file> [output-file]
#
# Hand it a packet. It reviews the packet's claims against the actual code in
# the working tree, grounded in the cpp-guidelines and cpp-performance MCP
# servers. Output is written to a file, not attached to a commit — the packet,
# not the commit, is the unit of work here.
#
# Default output: docs/research/<packet-name>-codex-review.md
#
# Reviewer independence is the point: this session's author should not be the
# only one judging whether the packet's claims hold.

set -euo pipefail

cd "$(dirname "$0")/.."

# Pinned here rather than in ~/.codex/config.toml so this script's behavior does
# not drift with the user's interactive default. Override with CODEX_MODEL.
MODEL="${CODEX_MODEL:-gpt-5.6-sol}"
EFFORT="${CODEX_EFFORT:-high}"

# The MCP servers this review is required to consult. Names must match the
# server keys in ~/.codex/config.toml (cpp-guidelines, cpp-performance) or the
# reviewer will cite tools it never called.
GUIDELINES_URL="${CPP_GUIDELINES_URL:-http://127.0.0.1:7011}"
PERF_URL="${CPP_PERF_URL:-http://127.0.0.1:7015}"

ARCH_CHECKLIST="docs/decisions/governance/cpp_architecture_review.md"
PERF_CHECKLIST="docs/decisions/governance/cpp_performance_review.md"

# ----------------------------------------------------------------------
# Preflight. Every check below fails loudly: a review that silently skips
# its guideline grounding is worse than no review, because it produces an
# artifact that looks like one.
# ----------------------------------------------------------------------

die() { echo "codex-review.sh: $*" >&2; exit 1; }

command -v codex >/dev/null 2>&1 || die "'codex' CLI not found on PATH"

PACKET="${1:-}"
OUTPUT_ARG="${2:-}"
[ -n "${PACKET}" ] || die "usage: $0 <packet-file> [output-file]"
[ -r "${PACKET}" ] || die "cannot read packet: ${PACKET}"

# The governance checklists live in a submodule. If it is not populated the
# review would silently lose both gates, so refuse rather than degrade.
for f in "${ARCH_CHECKLIST}" "${PERF_CHECKLIST}"; do
  [ -r "$f" ] || die "missing governance checklist: $f
  the governance submodule is not populated — run: git submodule update --init"
done

# The MCP servers are a hard requirement of this review, not a nice-to-have.
# Checking here converts a silent mid-review skip into an upfront failure.
check_mcp() {
  name="$1"; url="$2"
  # Any HTTP status means the server is listening. A bare GET returns 406
  # because the MCP streamable transport wants its own Accept headers, so
  # `curl -f` must NOT be used here — it would reject a healthy server.
  #
  # Branch on curl's exit status, not on the body. An earlier version used
  # `... || echo 000` as a fallback, but on connection failure curl BOTH
  # prints 000 and exits non-zero, producing "000000" and silently passing
  # the check this function exists to enforce. Covered by
  # tests/test_codex_review_preflight.sh.
  if ! curl -s -o /dev/null --max-time 5 "${url}" 2>/dev/null; then
    die "MCP server '${name}' is not reachable at ${url}
  This review is required to be grounded in it. Start the server and retry,
  rather than running a review that cannot check what it claims to check."
  fi
}
check_mcp "cpp-guidelines" "${GUIDELINES_URL}"
check_mcp "cpp-performance" "${PERF_URL}"

PACKET_BASE="$(basename "${PACKET}" .md)"
OUTPUT="${OUTPUT_ARG:-docs/research/${PACKET_BASE}-codex-review.md}"
[ -e "${OUTPUT}" ] && die "review already exists: ${OUTPUT}
  delete it or pass a different output path"

PROMPT_FILE="$(mktemp -t bllm-codex-prompt.XXXXXX)"
trap 'rm -f "${PROMPT_FILE}"' EXIT

# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------

{
  cat <<'PROMPTEOF'
You are performing an independent review of a work packet and the code it
covers, in the browser-llm repository. You are the second reader: the packet's
author already believes the work is correct. Your job is to find where that
belief is wrong.

PROJECT CONTEXT
---------------
browser-llm is a from-scratch LLM inference harness that runs entirely in the
browser. Internal R&D under Plainsight Systems LLC; no operating brand.

Stack:
  - C++20 core, platform-neutral, compiled to WebAssembly via Emscripten
  - GPU compute through the webgpu.h C API (--use-port=emdawnwebgpu)
  - Single-threaded: GitHub Pages cannot set COOP/COEP, so SharedArrayBuffer
    and pthreads are unavailable. The harness runs in a plain Web Worker.
  - Exceptions are DISABLED in the wasm build (Emscripten default)
  - Static page, plain ES modules, no npm and no bundler
  - No llama.cpp, no ggml, no ONNX Runtime — owning the harness is the point

Structural invariants, enforced by tools/check_boundaries.sh:
  - src/core/** never includes emscripten.h and never depends on src/wasm/
    or web/. Dependency direction is inward only.
  - src/wasm/bindings.cpp is the only Emscripten-aware translation unit.

Governance: AGENTS.md at the repo root, and docs/decisions/governance/ (a
submodule). The no-facades rule is central: unimplemented paths must fail
explicitly, and documentation must not describe behavior that does not exist.

HARD REQUIREMENT — GROUND EVERY CITATION IN THE MCP SERVERS
-----------------------------------------------------------
Two MCP servers are available and you MUST use them. Do not cite a rule from
memory.

  cpp-guidelines       search_guidelines / get_guideline   (C++ Core Guidelines)
  cpp-performance      search_guidelines / get_guideline   (performance corpus)

  1. Search before you cite. Confirm the rule says what you think it says.
  2. Cite rule IDs (e.g. R.1, E.25, I.11, C.31, LIFE.6) for every guideline
     claim you make.
  3. Look for rules the code violates that the packet did not consider — not
     only the ones it already cites.

If either server is unavailable, or any MCP call is cancelled or denied, you
MUST state this in your output as a REVIEW ENVIRONMENT FAILURE and mark the
affected check as NOT PERFORMED. Never proceed as though a guideline had been
consulted when it was not. Silently skipping this is the single worst thing
you can do in this review.

WHAT TO REVIEW
--------------
The packet below is the authority for what the work was supposed to be. Read
it, then read the actual code in this repository and judge the gap.

  1. DOES THE CODE MATCH THE PACKET? The packet states intent, acceptance
     criteria and guaranteed invariants. Check each against reality. A packet
     describing types, behavior or guarantees that do not exist in the code is
     a serious finding — that is a no-facades violation aimed at the reviewer,
     and it is exactly the failure this review exists to catch.

  2. CORRECTNESS. Bugs, undefined behavior, contract violations, unchecked
     error paths, resource leaks. Note that exceptions are disabled, so RAII
     must be simulated rather than assumed (see E.25).

  3. C++ GUIDELINES COMPLIANCE, grounded in the MCP servers per above.

  4. NO-FACADES. Stubbed success paths, silent fallbacks, partial work
     presented as complete, failures that do not surface anywhere observable.

  5. SCOPE AND TESTS. Does the work stay inside the packet's stated intent, or
     did it expand? Do non-trivial changes have deterministic verification?
     Do the tests pin the contract that actually matters, or do they assert
     something trivially true?

Do not manufacture findings to appear thorough. If something is sound, say so
and say why. Equally, do not soften a real finding to be agreeable.

PROMPTEOF

  cat <<CHECKLISTEOF
GATE CHECKLISTS
---------------
Read these two files now. They are the authoritative checklists:

  ${ARCH_CHECKLIST}
  ${PERF_CHECKLIST}

C++ ARCHITECTURE GATE — applies to any non-trivial C++ change. Evaluate the
core/wrapper boundary, component cohesion, dependency direction, header
discipline, interface design, ownership and lifetime, abstraction quality and
test surface. Use the checklist's own P0-P3 severity scale.

C++ PERFORMANCE GATE — decide from the checklist's own scope criteria whether
this work is performance-sensitive. This repo treats inference execution, model
load, memory footprint and GPU dispatch as performance-sensitive by default;
setup-time code that runs once is not. If it IS performance-sensitive, ground
your findings in the cpp-perf-guidelines MCP server. If it is NOT, say so
explicitly in one line and skip the gate — a stated null result is the correct
artifact, not an omission.

A P0 or P1 finding in either gate blocks acceptance.

OUTPUT FORMAT
-------------
Markdown. Lead with an outcome line:

  OUTCOME: approved | approved_with_notes | changes_requested | needs_decision

Then, as separate sections kept distinct from each other:
  - Summary (what the packet claimed, and whether it holds)
  - Findings — each with severity, file:line, why it matters, expected fix
  - C++ Architecture Review (checklist vocabulary and summary template)
  - C++ Performance Review, or one line stating why it does not apply
  - MCP grounding — which servers and tools you actually called, and any
    REVIEW ENVIRONMENT FAILURE
  - Residual risk

=== PACKET UNDER REVIEW: ${PACKET} ===
CHECKLISTEOF

  cat "${PACKET}"

  printf '\n=== END PACKET ===\n\n'
  printf 'Repository root is the current working directory. Read whatever\n'
  printf 'source, tests and build files you need to judge the claims above.\n'
} > "${PROMPT_FILE}"

# ----------------------------------------------------------------------
# Invoke codex
# ----------------------------------------------------------------------

# '-a on-request' is REQUIRED: the cpp-guidelines / cpp-perf-guidelines MCP
# tools are approval-gated in ~/.codex/config.toml. Under a bare 'codex exec'
# those calls are cancelled and the review proceeds having read no guideline
# at all. Do not drop this flag.
echo "codex-review.sh: reviewing ${PACKET}" >&2
echo "  model:  ${MODEL} (effort ${EFFORT})" >&2
echo "  output: ${OUTPUT}" >&2

mkdir -p "$(dirname "${OUTPUT}")"

codex -a on-request exec \
  -m "${MODEL}" \
  -c model_reasoning_effort="\"${EFFORT}\"" \
  -o "${OUTPUT}" \
  - < "${PROMPT_FILE}" || {
  RC=$?
  echo "codex-review.sh: codex exited non-zero (${RC})" >&2
  exit "${RC}"
}

[ -s "${OUTPUT}" ] || die "codex produced an empty review — not keeping it"

echo "codex-review.sh: review written to ${OUTPUT}" >&2

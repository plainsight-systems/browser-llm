#!/usr/bin/env bash
# codex-review.sh — independent review of a commit.
#
# Usage:  ./scripts/codex-review.sh [--post] <commit> [output-file]
#
# Hand it a commit. It reviews the commit's claims — its message, and the
# design its headers state — against the code, grounded in the cpp-guidelines
# and cpp-performance MCP servers. The commit is the unit of work
# (docs/decisions/workflow.md): a design commit is reviewed before it is
# implemented, an implementation commit against the design it carries out.
#
# The review is JSON in the shape of the schema below, written to the output
# file (default .cache/reviews/<sha>.json). With --post it is also posted as
# comments on the commit on GitHub (scripts/post_commit_review.py): one holding
# the whole review, and one on each finding's line in the diff.
#
# Reviewer independence is the point: this session's author should not be the
# only one judging whether the commit's claims hold.

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

POST=0
if [ "${1:-}" = "--post" ]; then POST=1; shift; fi
COMMIT_ARG="${1:-}"
OUTPUT_ARG="${2:-}"
[ -n "${COMMIT_ARG}" ] || die "usage: $0 [--post] <commit> [output-file]"
COMMIT="$(git rev-parse --verify --quiet "${COMMIT_ARG}^{commit}")" \
  || die "not a commit: ${COMMIT_ARG}"

# Comments anchor to the commit on GitHub, so it must be there.
if [ "${POST}" = 1 ]; then
  command -v gh >/dev/null 2>&1 || die "'gh' CLI not found on PATH; it posts the comments"
  [ -n "$(git branch -r --contains "${COMMIT}" 2>/dev/null)" ] \
    || die "commit ${COMMIT:0:7} is not on any remote branch — push it before posting a review of it"
fi

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

OUTPUT="${OUTPUT_ARG:-.cache/reviews/${COMMIT}.json}"
[ -e "${OUTPUT}" ] && die "review already exists: ${OUTPUT}
  delete it or pass a different output path"

PROMPT_FILE="$(mktemp -t bllm-codex-prompt.XXXXXX)"
SCHEMA_FILE="$(mktemp -t bllm-codex-schema.XXXXXX)"
trap 'rm -f "${PROMPT_FILE}" "${SCHEMA_FILE}"' EXIT

# ----------------------------------------------------------------------
# Output schema. Every finding names a file and, where it has one, a line of
# the commit's new version, so it can be posted on that line.
# ----------------------------------------------------------------------

cat > "${SCHEMA_FILE}" <<'SCHEMAEOF'
{
  "type": "object",
  "additionalProperties": false,
  "required": ["outcome", "summary", "findings", "architecture_review", "performance_review",
               "mcp_grounding", "residual_risk"],
  "properties": {
    "outcome": {"type": "string",
                "enum": ["approved", "approved_with_notes", "changes_requested", "needs_decision"]},
    "summary": {"type": "string"},
    "findings": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["severity", "path", "line", "title", "body", "guidelines"],
        "properties": {
          "severity": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},
          "path": {"type": "string"},
          "line": {"type": ["integer", "null"]},
          "title": {"type": "string"},
          "body": {"type": "string"},
          "guidelines": {"type": "array", "items": {"type": "string"}}
        }
      }
    },
    "architecture_review": {"type": "string"},
    "performance_review": {"type": "string"},
    "mcp_grounding": {"type": "string"},
    "residual_risk": {"type": "string"}
  }
}
SCHEMAEOF

# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------

{
  cat <<'PROMPTEOF'
You are performing an independent review of one commit in the Charlotte
repository. You are the second reader: the commit's author already believes
the work is correct. Your job is to find where that belief is wrong.

PROJECT CONTEXT
---------------
Charlotte is a from-scratch LLM inference harness that runs entirely in the
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

How work is done here (docs/decisions/workflow.md): the design of a change
lives in its file headers, which state the contract and the design and cite
the guidelines behind them by ID. A design commit is headers only, reviewed
before anything is implemented. Every optimization is labelled
"Optimization (browser)" or "Optimization (practice)" with its reason.

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
  3. Check every guideline the commit itself cites: does the rule say what
     the header claims it says, and does the design actually follow it?
  4. Look for rules the design violates that the commit did not consider —
     not only the ones it already cites.

If either server is unavailable, or any MCP call is cancelled or denied, you
MUST state this in your output as a REVIEW ENVIRONMENT FAILURE and mark the
affected check as NOT PERFORMED. Never proceed as though a guideline had been
consulted when it was not. Silently skipping this is the single worst thing
you can do in this review.

WHAT TO REVIEW
--------------
The commit message and its headers are the authority for what the work is
supposed to be. Run `git show <commit>` for the full change, read the code it
touches and whatever it depends on, and judge the gap.

  1. DOES THE DESIGN HOLD? For a design commit: is it correct, complete and
     implementable as stated? Do the headers agree with each other, with the
     existing code they depend on (residency/plan.h, weight_view.h, the GGUF
     reader), and with WebGPU's actual rules? A header promising behavior the
     interfaces cannot deliver is a serious finding.

  2. DOES THE CODE MATCH ITS CLAIMS? Every claim in the commit message and the
     headers — sizes, alignments, counts, what another project does — checked
     against reality.

  3. CORRECTNESS. Bugs, undefined behavior, contract violations, unchecked
     error paths, resource leaks, lifetimes. Exceptions are disabled, so RAII
     must be simulated rather than assumed (see E.25).

  4. C++ GUIDELINES COMPLIANCE, grounded in the MCP servers per above.

  5. NO-FACADES. Stubbed success paths, silent fallbacks, partial work
     presented as complete, failures that do not surface anywhere observable.

  6. SCOPE AND TESTS. Does the commit stay inside its stated intent? Will the
     design be testable deterministically, and does it say how?

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

OUTPUT
------
Answer in the JSON schema you were given:

  outcome              approved | approved_with_notes | changes_requested |
                       needs_decision
  summary              what the commit claims, and whether it holds
  findings             each with severity (P0-P3), the file path from the
                       repository root, the line in the commit's version of
                       that file (null if it concerns the file as a whole),
                       a short title, the body (why it matters, expected
                       fix), and the guideline IDs it rests on
  architecture_review  in the checklist's vocabulary and summary template
  performance_review   or one line stating why it does not apply
  mcp_grounding        which servers and tools you actually called, and any
                       REVIEW ENVIRONMENT FAILURE
  residual_risk

Write the text fields in Markdown. Each finding is posted on its line of the
commit, so its body must stand on its own.

=== COMMIT UNDER REVIEW: ${COMMIT} ===
CHECKLISTEOF

  git show --stat --format='%H%n%an <%ae>%n%ad%n%n%B' "${COMMIT}"

  printf '\n=== END COMMIT SUMMARY ===\n\n'
  printf 'Repository root is the current working directory. Run git show %s for\n' "${COMMIT}"
  printf 'the full diff, and read whatever source, tests and build files you need.\n'
} > "${PROMPT_FILE}"

# ----------------------------------------------------------------------
# Invoke codex
# ----------------------------------------------------------------------

# '-a on-request' is REQUIRED: the cpp-guidelines / cpp-perf-guidelines MCP
# tools are approval-gated in ~/.codex/config.toml. Under a bare 'codex exec'
# those calls are cancelled and the review proceeds having read no guideline
# at all. Do not drop this flag.
echo "codex-review.sh: reviewing ${COMMIT:0:7}" >&2
echo "  model:  ${MODEL} (effort ${EFFORT})" >&2
echo "  output: ${OUTPUT}" >&2

mkdir -p "$(dirname "${OUTPUT}")"

codex -a on-request exec \
  -m "${MODEL}" \
  -c model_reasoning_effort="\"${EFFORT}\"" \
  --output-schema "${SCHEMA_FILE}" \
  -o "${OUTPUT}" \
  - < "${PROMPT_FILE}" || {
  RC=$?
  echo "codex-review.sh: codex exited non-zero (${RC})" >&2
  exit "${RC}"
}

[ -s "${OUTPUT}" ] || die "codex produced an empty review — not keeping it"
python3 -c 'import json, sys; json.load(open(sys.argv[1]))' "${OUTPUT}" \
  || die "codex's review is not JSON — kept at ${OUTPUT} for inspection, not posted"

echo "codex-review.sh: review written to ${OUTPUT}" >&2

if [ "${POST}" = 1 ]; then
  python3 scripts/post_commit_review.py "${OUTPUT}" "${COMMIT}"
fi

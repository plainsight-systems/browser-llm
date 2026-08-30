#!/usr/bin/env bash
# codex-review.sh — non-interactive Codex code review attached as git note.
#
# Usage:  ./scripts/codex-review.sh [<commit-sha>]
#         (default: HEAD)
#
# Per docs/decisions/workflow.md:
#   - Codex reviews every dev packet's commit
#   - Output is captured as a git note attached to the commit (W-2a)
#   - Codex is instructed to cross-check the C++ Core Guidelines
#     citations the developer claims in the commit message body (Layer C
#     of the cpp-guidelines triple-layer enforcement)
#   - For commits that touch C++ source, the review additionally drives
#     the C++ architecture + performance gate checklists
#     (docs/decisions/cpp_architecture_review.md +
#     cpp_performance_review.md; AGENTS.md §3.10 / workflow.md W-14)
#   - For commits that touch architecture packets (A{number}-*.md in
#     docs/decisions/packets/), the review additionally applies the
#     workflow.md "Architecture Work" section criteria (artifact
#     citations, facts/inferences/opinions discipline, handoff
#     readability, non-recommendations, supersession)
#
# Notes live in refs/notes/commits namespace. View with:
#   git notes show <commit-sha>
#   git log --notes=commits
#
# Notes are NOT pushed to remotes by default. To push:
#   git push origin refs/notes/commits

set -euo pipefail

# ----------------------------------------------------------------------
# Sanity / arguments
# ----------------------------------------------------------------------

if ! command -v codex >/dev/null 2>&1; then
  echo "codex-review.sh: 'codex' CLI not found on PATH" >&2
  echo "  Install: see https://github.com/openai/codex-cli" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "codex-review.sh: 'git' not found on PATH" >&2
  exit 1
fi

cd "$(dirname "$0")/.."

COMMIT_REF="${1:-HEAD}"
COMMIT_SHA="$(git rev-parse "${COMMIT_REF}" 2>/dev/null || true)"
if [ -z "${COMMIT_SHA}" ]; then
  echo "codex-review.sh: cannot resolve commit ref: ${COMMIT_REF}" >&2
  exit 1
fi
COMMIT_SHORT="$(git rev-parse --short "${COMMIT_SHA}")"

# Detect whether this commit touches C++ source. If so, the review
# additionally drives the C++ architecture + performance gate
# checklists (AGENTS.md §3.10 / workflow.md W-14) — see the conditional
# prompt section below.
CPP_TOUCHED="$(git show --name-only --format= "${COMMIT_SHA}" \
  | grep -E '\.(cpp|cc|cxx|h|hpp|hxx|mm)$' || true)"

# Detect whether this commit touches an architecture packet
# (A{number}-*.md in docs/decisions/packets/). If so, the review
# additionally applies the workflow.md "Architecture Work" section
# criteria — see the conditional prompt section below.
ARCH_TOUCHED="$(git show --name-only --format= "${COMMIT_SHA}" \
  | grep -E '^docs/decisions/packets/A[0-9]+-[^/]*\.md$' || true)"

# Refuse to overwrite an existing note silently — caller can delete the
# note explicitly if they want a fresh review.
if git notes list 2>/dev/null | grep -q "^[0-9a-f]\+ ${COMMIT_SHA}$"; then
  echo "codex-review.sh: note already exists for ${COMMIT_SHORT}" >&2
  echo "  view:   git notes show ${COMMIT_SHA}" >&2
  echo "  remove: git notes remove ${COMMIT_SHA}" >&2
  exit 1
fi

# ----------------------------------------------------------------------
# Build the review prompt
# ----------------------------------------------------------------------

PROMPT_FILE="$(mktemp -t vigil-codex-prompt.XXXXXX)"
REVIEW_FILE="$(mktemp -t vigil-codex-review.XXXXXX)"
trap 'rm -f "${PROMPT_FILE}" "${REVIEW_FILE}"' EXIT

# Heredoc expands $COMMIT_SHA inside the bash string. The git-show
# block is interpolated below the heredoc to avoid escaping issues.
{
  cat <<EOF
You are performing a code review on a Vigil commit. Output a structured
review. Be specific: file:line references; cite C++ Core Guidelines rule
IDs; concrete suggestions, not vague advice.

PROJECT CONTEXT
---------------
Vigil is a CustodyZero product: lifelong local personal intelligence on
Apple Silicon (v1-demo) then cross-platform (v1-launch).

Stack (locked):
  - C++23 harness core
  - MLX inference backend on Apple Silicon (Gemma 4 26B-A4B 4-bit)
  - Apple-only at v1-demo; cross-platform v1-launch later
  - SwiftUI shell on macOS / iOS, Qt on Linux, C++/WinRT on Windows
  - No Python in shipping product, no Rust, no telemetry

Operator profile: AGENTS.md at the repo root. The user-global CLAUDE.md
AI Operator Profile applies (CLAUDE.md §3 No-Facades, §3.5 tests required,
§5 intent must be explicit, §5.2 single intent per change, §5.6 no scope
expansion).

HARD REQUIREMENT — C++ CORE GUIDELINES
---------------------------------------
Per AGENTS.md §3.4, every C++ commit must show evidence of cpp-guidelines
querying in the message body — i.e., the developer must cite the rule
IDs they consulted (e.g., "ES.20", "F.16", "R.20", "C.21", etc.).

TOOLING — USE THE cpp-guidelines MCP SERVER FIRST (for any C++ review):
The authoritative source is the local cpp-guidelines MCP server. For a
C++ review you MUST consult it FIRST to ground every rule citation: call
its search tool (cpp-guidelines / search_guidelines) and, for a specific
rule, its lookup tool (cpp-guidelines / get_guideline) BEFORE relying on
memory. The public C++ Core Guidelines URL is a FALLBACK ONLY, for when
the MCP server is unavailable:
  https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines

If the cpp-guidelines MCP server or its tools are unavailable, or any MCP
tool call is cancelled or denied, you MUST state this explicitly in your
output as a REVIEW ENVIRONMENT FAILURE and treat the MCP-grounded
cross-check as NOT performed. Do NOT silently proceed as if the
guidelines had been checked. (If this commit touches no C++ source, there
are no rule IDs to cross-check — say so in one line and skip this section,
with no environment-failure claim.)

Your review tasks for cpp-guidelines specifically:
  1. Check that the developer's commit message body cites rule IDs.
     If absent, that is itself a workflow violation — flag it.
  2. Cross-check the cited rules against the actual code. If the
     developer claimed rule X but the code violates X, flag it.
  3. Check for unstated rules the code violates that the developer
     should have caught (e.g., obvious RAII / ownership / const-correctness
     issues that the relevant guideline category covers).
  4. Confirm rule wording via the cpp-guidelines MCP tools (search/get);
     use the URL above only as a fallback if the MCP server is unavailable.

REVIEW SCOPE
------------
Provide a review covering:
  - Correctness: bugs, undefined behavior, contract violations.
  - cpp-guidelines compliance (per the section above).
  - CLAUDE.md §3 No-Facades violations: stubbed-success paths, silent
    fallbacks, partial implementations dressed as complete.
  - Scope discipline (CLAUDE.md §5.2, §5.6): does the diff stay inside
    the stated intent? Any "while we're here..." territory?
  - Test discipline (CLAUDE.md §3.5, §7): non-trivial changes must have
    deterministic verification. Mocks must not erase the contract being
    tested.
  - Build + style: nothing obviously broken; reasonable C++23.

FORMAT
------
Structured. Headings for each finding. Severity (must-fix / should-fix /
nit). file:line references. Cite C++ Core Guidelines rule IDs where
relevant.

If the code is sound, say so explicitly and explain why (one paragraph).
Do not manufacture issues to seem thorough. Do not hand-wave dismiss
real issues to be agreeable.

EOF

  # ----------------------------------------------------------------
  # Coordinator-supplied authoritative rule bodies. The codex
  # environment's get_guideline MCP lookup is blocked (invalid_grant;
  # only search_guidelines succeeds — QUEUE.md review-process note). To
  # ground the cross-check against actual rule TEXT rather than search
  # snippets, the coordinator (who DOES have cpp-guidelines /
  # cpp-performance MCP access) fetches the relevant rule bodies and
  # passes their file path via VIGIL_CPP_GUIDELINE_BODIES. Independence
  # is preserved: codex still judges the diff against the rules itself;
  # only the rule text is supplied. Codex is told to disclose that it
  # relied on supplied bodies rather than its own lookup.
  if [ -n "${VIGIL_CPP_GUIDELINE_BODIES:-}" ] && [ -r "${VIGIL_CPP_GUIDELINE_BODIES}" ]; then
    cat <<'BODYEOF'

AUTHORITATIVE RULE BODIES (coordinator-fetched)
-----------------------------------------------
The rule bodies below were fetched from the local cpp-guidelines /
cpp-performance MCP servers by the coordinator at review time and are
authoritative. Use them to ground your cross-check. Because your own
get_guideline MCP call is environment-blocked, these bodies substitute
for it — but you MUST state in your output that you relied on
coordinator-supplied rule bodies rather than your own get_guideline
lookup (search_guidelines, if available to you, is still an independent
check you should run). Do NOT claim you independently fetched these.
BODYEOF
    cat "${VIGIL_CPP_GUIDELINE_BODIES}"
    printf '\n'
  fi

  # ----------------------------------------------------------------
  # C++ architecture + performance gate cross-check. Appended only
  # when the commit touches C++ source (AGENTS.md §3.10 / workflow.md
  # W-14). The checklist files are governance symlinks in
  # docs/decisions/ — Codex reads them directly so the checklists
  # stay single-source (a governance update is picked up on the next
  # review with no change to this script).
  # ----------------------------------------------------------------
  if [ -n "${CPP_TOUCHED}" ]; then
    cat <<'CPPEOF'

HARD REQUIREMENT — C++ ARCHITECTURE + PERFORMANCE GATES
--------------------------------------------------------
This commit touches C++ source. Beyond the rule-level cpp-guidelines
cross-check above, AGENTS.md §3.10 and workflow.md W-14 require two
gate-level reviews for C++ changes. These gates catch "agentic
decomposition" — code that passes rule-level C++ checks while making
weak choices at the file / component / module / hot-path level.

1. C++ ARCHITECTURE REVIEW — applies to every non-trivial C++ change.
   Read the checklist file now:  docs/decisions/cpp_architecture_review.md
   It is the authoritative checklist (a governance symlink). Apply it:
   evaluate the core/wrapper boundary, physical components, dependency
   direction, header discipline, interface design, ownership + lifetime,
   abstraction quality, and test surface. Use its P0-P3 severity scale.

2. C++ PERFORMANCE REVIEW — applies only when the change is
   performance-sensitive: inference hot paths, the decode loop /
   sampler / KV-cache / forward path, real-time loops, high-volume
   data, startup / model-load time, or memory-sensitive code.
   Read the checklist file now:  docs/decisions/cpp_performance_review.md
   Decide from its own scope criteria whether this change is
   performance-sensitive. If it IS, you MUST consult the local
   `cpp-performance` MCP server FIRST to ground any performance-guidance
   citation: its search tool (`cpp-performance` / `search_guidelines`)
   and, for a specific item, its lookup tool (`cpp-performance` /
   `get_guideline`); public performance docs are a FALLBACK ONLY. Then
   apply the checklist: budget + measurement, hot paths, data layout,
   allocation, copies, concurrency, edge-AI runtime configuration. If the
   `cpp-performance` MCP server/tool is unavailable, or an MCP call is
   cancelled or denied, state this explicitly as a REVIEW ENVIRONMENT
   FAILURE and treat the performance-guidance check as NOT performed — do
   not proceed as if it had been checked. If it is NOT
   performance-sensitive, say so explicitly in one line and skip it.

A P0 or P1 architecture or performance finding BLOCKS acceptance.

In your output, add a dedicated "C++ Architecture Review" section
(and, when applicable, a "C++ Performance Review" section) using the
outcome vocabulary and summary template from the respective checklist
file, in addition to the rule-level findings above. Keep the
architecture/performance findings distinct from the cpp-guidelines
rule-level findings — they are separate gates.
CPPEOF
  fi

  # ----------------------------------------------------------------
  # Architecture packet review criteria. Appended only when the commit
  # touches one or more A{number}-*.md architecture packets
  # (workflow.md "Architecture Work" section). The criteria are
  # workflow-level and do not require source files to be referenced —
  # A-packets do not modify code.
  # ----------------------------------------------------------------
  if [ -n "${ARCH_TOUCHED}" ]; then
    cat <<'ARCHEOF'

HARD REQUIREMENT — ARCHITECTURE PACKET REVIEW
---------------------------------------------
This commit touches one or more architecture packets (A{number}-*.md).
Beyond the standard scope/no-facades/correctness review above, per the
workflow.md "Architecture Work" section, architecture packets must be
audited for:

1. ARTIFACT CITATIONS — every fact claim must cite a verifiable artifact
   (packet number, run directory, or source file:line). Inference claims
   must be marked as such and back-reference the evidence they build on.
   Flag any unmarked inference-as-fact, and any claim without a citation
   that the reader could not verify from the named artifact.

2. FACTS / INFERENCES / OPINIONS DISCIPLINE — findings sections must
   explicitly mark each claim's class (fact, inference, or opinion) and
   confidence level (high, medium, low). Unmarked claims block
   acceptance. Confidence levels must be calibrated to the evidence: a
   claim citing one observation should not be marked "high confidence."

3. HANDOFF READABILITY — recommended-next-packets must be self-contained
   for the implementing role. Each must state intent, prerequisites,
   success criteria, and any residual risk to declare. A reader without
   the originating conversation should be able to shape the named
   implementation packet from the recommendation alone.

4. NON-RECOMMENDATIONS — an explicit "do not do X, because Y" section is
   required for any tempting path the architecture packet rules out.
   Missing this section, or vague non-recommendations without rationale,
   are findings.

5. SUPERSESSION — predecessor and successor links are required even if
   "none yet." Stale supersession (e.g., a packet still marked draft
   after its successor lands) is a finding.

A must-fix finding in any of the above blocks acceptance.

In your output, add a dedicated "Architecture Packet Review" section
using the same severity vocabulary (must-fix / should-fix / nit) as the
rest of the review, in addition to the standard findings above. Keep
the architecture-packet findings distinct from the standard findings —
they are separate gates.
ARCHEOF
  fi

  printf '\n=== COMMIT MESSAGE ===\n'
  git log -1 --format=%B "${COMMIT_SHA}"

  printf '\n=== DIFF ===\n'
  git show --no-color "${COMMIT_SHA}"
} > "${PROMPT_FILE}"

# ----------------------------------------------------------------------
# Invoke codex
# ----------------------------------------------------------------------

# codex CLI 0.125+ uses 'codex exec' for general non-interactive mode.
# We tried 'codex review --commit <SHA>' but its argument parser rejects
# a custom PROMPT alongside --commit (the usage line claims to accept it,
# the parser disagrees — Codex CLI bug or doc drift). 'codex exec' is the
# robust path: we assemble the diff + message + cross-check instructions
# ourselves and pass them via stdin.
#
# 'codex exec -' reads instructions from stdin per --help.
#
# '-a on-request' (top-level approval policy) is REQUIRED for the MCP
# cross-checks. The cpp-guidelines / cpp-performance MCP tools are
# approval-gated in ~/.codex/config.toml (search_guidelines
# approval_mode = "approve"). Under bare 'codex exec -' those MCP calls
# are cancelled ("user cancelled MCP tool call") and the review silently
# proceeds without ever reading a guideline. Running through the
# on-request approval policy lets the approval-gated MCP tools execute
# (probe-confirmed: cpp-guidelines -> E.6, cpp-performance -> CACHE.6).
# Do not drop this flag (packet 246).

echo "codex-review.sh: invoking codex exec (-a on-request) on ${COMMIT_SHORT} ..." >&2
codex -a on-request exec - < "${PROMPT_FILE}" > "${REVIEW_FILE}" || {
  RC=$?
  echo "codex-review.sh: codex exited non-zero (${RC})" >&2
  echo "codex-review.sh: review output (partial) follows:" >&2
  cat "${REVIEW_FILE}" >&2 || true
  exit "${RC}"
}

if [ ! -s "${REVIEW_FILE}" ]; then
  echo "codex-review.sh: codex produced empty review — refusing to attach empty note" >&2
  exit 1
fi

# ----------------------------------------------------------------------
# Attach as git note
# ----------------------------------------------------------------------

git notes add -F "${REVIEW_FILE}" "${COMMIT_SHA}"

echo "codex-review.sh: review attached to ${COMMIT_SHORT}" >&2
echo "  view:   git notes show ${COMMIT_SHA}" >&2
echo "  log:    git log --notes=commits -1 ${COMMIT_SHA}" >&2
echo "  remove: git notes remove ${COMMIT_SHA}" >&2

# Workflow

This repo uses Lite Factory workflow from `governance/product_memory_workflow.md`.

## Default Chain

```text
Coordinator
  -> packet
  -> implementation
  -> review
  -> QA
  -> MEMORY.md and QUEUE.md updates
```

## Local Notes

This repo is C++-dominant and performance-sensitive, so the default chain is
extended at two points:

```text
packet          -> must carry a C++ architecture note for non-trivial C++ work,
                   and a C++ performance note for performance-sensitive work
review          -> C++ architecture review (governance/cpp_architecture_review.md)
                -> C++ performance review (governance/cpp_performance_review.md)
```

- **All C++ work must be checked against the `cpp-guidelines` MCP server
  before implementation, and all performance-sensitive work against
  `cpp-perf-guidelines`.** Record the check in the packet's review record,
  including when it produces no material findings — "checked, nothing
  material, because this is setup-time code" is the artifact, not an
  omission. If either server is unreachable, say so and get explicit
  agreement before writing C++ rather than proceeding quietly.
- **Never retrofit a packet for work that is already complete.** A packet
  gates work before it starts; written afterwards it changes nothing and only
  makes the process look followed. "This shipped without a packet" is a lesson
  about sequencing — record it once and move on.
- Acceptance is blocked on any P0 or P1 architecture or performance finding.
- Treat inference execution, model load, memory footprint, and GPU dispatch as
  performance-sensitive by default. A packet touching those paths without a
  performance note is not ready.
- A baseline measurement must exist before an optimization packet is accepted.
  "Faster" without a before/after number is not verification.
- Browser and GPU behavior is environment-sensitive. Verification must name the
  blessed targets it ran on; a green build is not proof.

Do not weaken inherited governance without a decision.

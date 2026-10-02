# Workflow

This repo adapts the Lite Factory workflow from
`governance/product_memory_workflow.md`. The design of each change lives in
the code it changes, not in a separate packet.

## Default Chain

```text
Coordinator
  -> file headers: the contract, the design, and the guidelines behind it
  -> review of the headers
  -> implementation and tests
  -> review
  -> MEMORY.md and QUEUE.md updates
```

## Local Notes

This repo is C++-dominant and performance-sensitive.

- **The headers are the plan.** Each new or changed file's header states its
  contract and its design, and cites by ID the guidelines from the
  `cpp-guidelines` and `cpp-perf-guidelines` corpora that shape it (for
  example C.2, E.4, CACHE.3, EMB.6). The headers are written and reviewed
  before the implementation.
- **Optimizations are designed in, not deferred.** The harness exists to show
  both the optimizations the browser forces and the ones production systems
  use at scale. Each carries `// Optimization (browser): …` or
  `// Optimization (practice): …`, with its concrete reason and the guideline
  or system it follows, so a reader sees a choice rather than an accident.
  Where it can be measured, it is, and the figures go in the note.
- **Every C++ step is checked against `cpp-guidelines` before its code is
  written, and performance-sensitive work against `cpp-perf-guidelines`.** The
  citations in the headers and optimization notes are the record. If either
  server is unreachable, say so and get explicit agreement before writing C++.
- Acceptance is blocked on any P0 or P1 finding from the C++ architecture
  review (`governance/cpp_architecture_review.md`) or the C++ performance
  review (`governance/cpp_performance_review.md`).
- Inference execution, model load, memory footprint and GPU dispatch are
  performance-sensitive by default.
- Browser and GPU behavior is environment-sensitive. Verification names the
  blessed targets it ran on; a green build is not proof.
- `packets/` holds the records of BLLM-001 to BLLM-003. New work does not
  start with a packet.

Do not weaken inherited governance without a decision; MEMORY.md records the
ones taken.

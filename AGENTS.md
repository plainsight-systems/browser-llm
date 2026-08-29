# Agent Instructions

This repository inherits Plainsight Systems governance from `docs/decisions/`.

## Session Entry

1. Read `docs/decisions/MEMORY.md`.
2. Read `docs/decisions/QUEUE.md`.
3. Read `docs/decisions/workflow.md`.
4. Read inherited governance docs relevant to the task.
5. Inspect product-specific source and tests before editing.

## Operating Rules

- Follow Lite Factory workflow from `docs/decisions/product_memory_workflow.md`.
- No non-trivial implementation without a packet in `docs/decisions/packets/`.
- Keep product decisions in `docs/decisions/` and research in `docs/research/`.
- Update `MEMORY.md` and `QUEUE.md` when work state or durable knowledge changes.
- Do not weaken inherited governance without an explicit decision.

## Repo-Specific Gates

This repo is C++-dominant and performance-sensitive. Both C++ gates bind:

- Non-trivial C++ changes require a C++ architecture note in the packet before
  implementation, and a review against `docs/decisions/cpp_architecture_review.md`
  before acceptance.
- Performance-sensitive C++ changes require a C++ performance note in the packet
  before implementation, and a review against
  `docs/decisions/cpp_performance_review.md` before acceptance.
- Treat inference execution paths, model load, memory footprint, and GPU
  dispatch as performance-sensitive by default.

## Scope Posture

`engineering_philosophies.md` forbids demo-ware. In this repo, "tech demo" means
the real inference harness on a deliberately narrow slice at full intended
quality. It does not mean a lower-fidelity preview to be hardened later.

- Narrow the slice rather than lowering the quality inside it.
- A contract relaxation must be defensible without using the word "demo."

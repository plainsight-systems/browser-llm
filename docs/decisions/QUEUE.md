# Work Queue

This file tracks active and accepted work.

## Active

- **BLLM-001: Repo skeleton and build system.** Packet:
  `packets/2026-08-29-repo-skeleton-and-build-system.md`. Status
  `changes_requested` after independent codex review
  (`research/2026-08-29-repo-skeleton-and-build-system-codex-review.md`).
  Implementation is complete and deployed; the review's P1 and P2 findings
  have been worked. Second review completed 2026-08-30 with a full guideline
  audit (no environment failure) and again returned `changes_requested`;
  `research/2026-08-30-bllm-001-codex-review-2.md`. Its P1 and the mechanical
  P2/P3 findings are fixed. Two design-level findings remain open and are
  tracked below rather than half-fixed.

- **BLLM-005: Async callback contracts and wrapper state.** From the second
  review. Callback non-null preconditions are neither stated nor checked;
  `RequestCallback` passes an error pointer into a temporary with undocumented
  lifetime; the bridging contexts use naked `new` / `delete this` outside the
  `UniqueHandle` system; and `active_device()` / `request_in_flight()` remain
  mutable singleton state that construct-on-first-use does not eliminate
  (I.5, I.12, ES.65, R.11, ES.24, I.2, R.6). Needs owned result objects and
  run state threaded through userdata, plus fake-async tests covering every
  completion path.

## Ready

- **BLLM-002: Model, quantization and blessed targets.** Not yet written.
  Settles target model, parameter count, quantization format, weight layout,
  and the performance budget with a baseline. Must declare a **limits floor**
  — observed limits far exceed the spec minimums, so weight residency is a
  portability decision (`research/2026-08-29-webgpu-limits-observed.md`).
  Change class: architectural. Requires both C++ notes.

- **BLLM-003: Native Dawn for GPU-kernel tests.** Lands with the first model
  kernel. Until then `device.cpp` and `self_check.cpp` compile for wasm only
  and no kernel has a deterministic native oracle.

## Accepted

- None.

## Parking Lot

- None.

# Work Queue

This file tracks active and accepted work.

## Active

- **BLLM-001: Repo skeleton and build system.** Packet:
  `packets/2026-08-29-repo-skeleton-and-build-system.md`. Status
  `changes_requested` after independent codex review
  (`research/2026-08-29-repo-skeleton-and-build-system-codex-review.md`).
  Implementation is complete and deployed; the review's P1 and P2 findings
  have been worked. Remaining before acceptance: re-run the review now that
  `get_guideline` approval is configured, since the first pass recorded the
  guideline audit as NOT PERFORMED.

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

- **BLLM-004: Review tooling.** `scripts/codex-review.sh` was added outside any
  packet, which the review correctly flagged against one-packet/one-intent.
  Retrofit a packet covering it and its two guard tests.

## Accepted

- None.

## Parking Lot

- None.

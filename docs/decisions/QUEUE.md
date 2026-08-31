# Work Queue

This file tracks active and accepted work.

## Active

- **BLLM-002: Model selection and weight residency.** Packet:
  `packets/2026-08-31-model-selection-and-weight-loading.md`. Status: draft,
  awaiting review before implementation. Decides Qwen3-0.6B / Q4_0 / GGUF, and
  gets the weights onto the GPU **still quantized**, proven byte-identical by
  readback. Scoped around the highest-risk unknown — packing ~420 MB across
  buffers under a limit granted at runtime — rather than around what was
  easiest to test.

## Ready

- **BLLM-003: Native Dawn for GPU-kernel tests.** Lands with the first model
  kernel — the first point at which a kernel's correctness is not
  self-evident and needs a CPU reference oracle. Until then `device.cpp` and
  `self_check.cpp` compile for wasm only, and the GPU path is verified solely
  by the page's readback assertion.

## Accepted

- **BLLM-001: Repo skeleton and build system** — accepted 2026-08-31.
  `packets/2026-08-29-repo-skeleton-and-build-system.md`.

  Dual-target CMake build, platform-neutral core, WebGPU device path, wasm
  bindings, and a static page deployed by CI. Two independent reviews by a
  separate identity, both `changes_requested`, both sets of findings worked.
  Residual risk is recorded in the packet's acceptance section — chiefly that
  no native GPU test exists until BLLM-003.

## Parking Lot

- None.

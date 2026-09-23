# Work Queue

This file tracks active and accepted work.

## Active

- **BLLM-002: GGUF reading and Q4_0 layout.** Packet:
  `packets/2026-08-31-model-selection-and-weight-loading.md`. Decides
  Qwen3-0.6B / Q4_0 / GGUF and delivers the container reader plus the Q4_0
  block layout and its bit-exact oracle.

  Scope was cut back on 2026-09-23 to what was actually built. The packet
  originally carried twelve criteria; the eight covering tokenizer, config,
  residency, upload and reporting were planning written far ahead of the code
  and are removed rather than carried. Forward work is unplanned and will be
  scoped when it is started.

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

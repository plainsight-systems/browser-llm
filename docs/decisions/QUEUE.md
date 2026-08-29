# Work Queue

This file tracks active and accepted work.

## Active

- **BLLM-001: Repo skeleton and build system.** Packet:
  `packets/2026-08-29-repo-skeleton-and-build-system.md`. Change class:
  architectural. Build system, platform-neutral core, native test suite,
  boundary checks and native CI are implemented and verified. Remaining:
  `core/gpu` device/buffer/pipeline, `src/wasm/bindings.cpp`, `web/`, and
  enabling the Pages workflow on push. Blocked on a working Emscripten
  toolchain to compile against — the WebGPU device path will not be written
  unverified.

## Ready

- **BLLM-002: Model, quantization and blessed targets.** Not yet written.
  Settles target model, parameter count, quantization format, weight layout,
  and the performance budget with a baseline. Change class: architectural.
  Requires a C++ architecture note and a C++ performance note. Nothing in
  BLLM-001 depends on it.

## Accepted

- None.

## Parking Lot

- None.

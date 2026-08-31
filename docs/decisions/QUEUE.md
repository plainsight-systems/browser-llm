# Work Queue

This file tracks active and accepted work.

## Active

- None.

## Ready

- **BLLM-002: Model, quantization and blessed targets.** Not yet written.
  Settles target model, parameter count, quantization format, weight layout,
  and the performance budget with a baseline.

  Must declare a **limits floor**. The device is granted WebGPU's defaults
  unless `requiredLimits` asks for more; this repo now asks for the adapter's
  advertised maxima, which is always satisfiable but means limits **vary per
  device and are known only after acquisition**. Weight residency must be
  planned from the granted limits rather than a compile-time constant, and
  must still degrade sensibly on an adapter that advertises only the defaults.
  See `research/2026-08-29-webgpu-limits-observed.md`, which records three
  successive wrong conclusions about this and why each was wrong.

  Change class: architectural. Requires a C++ architecture note and a C++
  performance note.

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

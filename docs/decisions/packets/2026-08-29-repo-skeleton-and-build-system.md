# BLLM-001: Repo skeleton and build system

**Status:** review_requested
**Change class:** architectural

## Intent

- **What is changing:** The repository gains its physical shape and build
  system. A dual-target CMake build produces (a) a native binary and test
  suite from platform-neutral C++ and (b) a WebAssembly module for the
  browser, from the same core sources. A static page loads that module in a
  Web Worker, acquires a WebGPU device through the `webgpu.h` C API, runs one
  trivial compute shader, and verifies the readback. CI builds and tests
  natively on every push and deploys the page to GitHub Pages.

- **Why the change is necessary:** No implementation can begin without a
  toolchain, and the toolchain choices here are not reversible cheaply. The
  boundary between platform-neutral core C++ and the browser wrapper, and the
  decision to reach the GPU through `webgpu.h` rather than JavaScript,
  determine whether inference code can be tested outside a browser at all.
  Establishing them before any model code exists is the difference between a
  testable harness and one that is only observable through a page.

- **Expected behavior changes:** The repository builds where it previously
  contained no source. `make test` produces a native binary that runs unit
  tests without a browser or GPU. `make wasm` produces an ES module and its
  `.wasm`. The deployed page reports the WebGPU adapter, its limits, and the
  result of a vector-add readback. Nothing infers anything; no model is
  loaded; no weights are fetched.

- **Guaranteed invariants/contracts:**
  - `src/core/**` never includes `emscripten.h` and never references anything
    under `src/wasm/` or `web/`. Dependency direction is one-way.
  - `src/wasm/bindings.cpp` is the only Emscripten-aware translation unit.
  - The same `src/core` sources compile in both the native and wasm
    configurations. No `#ifdef __EMSCRIPTEN__` in core.
  - The build never requires `SharedArrayBuffer`, cross-origin isolation, or
    pthreads. GitHub Pages cannot set COOP/COEP headers, so any dependency on
    them would break deployment.
  - WGSL shader sources are single-sourced. The same file is compiled into
    both targets by build-time codegen; there is no second copy.

## Acceptance Criteria

- [ ] `cmake --preset native-debug && cmake --build --preset native-debug`
      succeeds from a clean tree.
- [ ] `ctest --preset native-debug` passes, exercising real logic, without a
      browser and without a GPU.
- [ ] The wasm configuration builds inside the pinned Emscripten image and
      emits `browser_llm.mjs` and `browser_llm.wasm`.
- [ ] Grep check passes: no file under `src/core/` matches `emscripten.h`,
      `EM_JS`, `EMSCRIPTEN_`, or `#include "wasm/`.
- [ ] Grep check passes: `src/wasm/bindings.cpp` is the only file under `src/`
      that includes `emscripten.h`.
- [ ] Shader embedding is verified: the generated C++ constant is byte-identical
      to `src/shaders/vector_add.wgsl`, asserted by a native test rather than
      by inspection.
- [ ] Serving `dist/` over http and opening the page in Chrome reports a
      non-empty adapter description, the four buffer limits this project is
      constrained by, and `vector_add: OK`.
- [ ] Both CI workflows pass on a push.
- [ ] `README.md` documents the layout as built, with planned-but-absent
      directories marked as planned.

## Verification Plan

Native unit tests cover the logic that exists at this stage and no more:

| Test | What it pins |
|---|---|
| `dispatch_math_test` | Workgroup-count arithmetic: exact multiples, non-multiples, zero elements, and sizes near the `maxComputeWorkgroupsPerDimension` bound. Pure function, exhaustively checkable. |
| `shader_embed_test` | The embedded shader constant equals the on-disk `.wgsl` byte for byte, so the single-source invariant cannot silently break. |

Structural invariants are verified by grep in CI rather than by review, because
a reviewer noticing a stray `#include <emscripten.h>` is not a control.

The GPU path is verified in-browser at this stage by readback assertion. A
vector-add is self-evidently correct or not; it needs no oracle. This is
deliberately the last packet where that is true.

**Explicitly not verified here, and why:** there is no CPU-side model logic
yet, so there is nothing further to unit test. Build success is normally not
evidence of correctness, but for a packet whose deliverable *is* the build,
the build succeeding under a pinned toolchain, plus a real GPU readback, is
the contract. Unit-test surface arrives with BLLM-002 and its GGUF parsing.

## C++ Architecture Note

- **Core components touched:** New. `core/gpu` only — device acquisition,
  buffer creation and readback, compute pipeline construction, dispatch
  arithmetic, and a self-check that runs the vector-add end to end.

- **Platform wrappers touched:** New. `src/wasm/bindings.cpp` exposes the
  self-check and device description to JavaScript. `web/` owns the page, the
  worker, and rendering.

- **Public interfaces changed:** None exist yet. Introduced: `gpu::Device`,
  `gpu::Buffer`, `gpu::ComputePipeline`, `gpu::dispatch_count`, and
  `gpu::run_self_check`.

- **New files/classes/targets/modules:** Targets `browser_llm_core` (static
  library, both configurations), `browser_llm_tests` (native only),
  `browser_llm` (wasm only).

- **Dependency edges added or removed:** `browser_llm_core` depends on
  `webgpu.h` only. In the wasm configuration that header and its
  implementation come from `--use-port=emdawnwebgpu`; natively they would come
  from Dawn, which is not linked in this packet. `browser_llm_tests` depends
  on core and doctest. `browser_llm` depends on core and Emscripten. No edge
  points from core outward.

- **Ownership/lifetime model:** RAII throughout. `Device` owns the WGPU
  instance, adapter, and device handles and releases them in reverse
  acquisition order. `Buffer` and `ComputePipeline` own single WGPU handles
  and are move-only, non-copyable — a copied handle would double-release.
  Nothing is shared; no `shared_ptr` appears.

- **Test surface independent of UI wrappers:** `dispatch_count` and the shader
  embedding are pure and tested natively. `Device`, `Buffer`, and
  `ComputePipeline` are written against `webgpu.h` specifically so they can be
  linked against native Dawn later without modification. That capability is
  established here and exercised in the first kernel packet.

- **Why this decomposition is appropriate:** The core/wrapper split is the
  only structural decision that is expensive to reverse. Everything else in
  this packet is a build file. Placing WebGPU behind `webgpu.h` inside core,
  rather than in the JS layer, is what keeps the forward pass — when it exists
  — in testable C++ instead of split across the WASM boundary.

- **What is intentionally not abstracted:** There is no rendering, tensor, or
  model abstraction. No interface has more than one implementation and none is
  introduced speculatively. `Device` is a concrete type, not an interface;
  substitutability is provided by `webgpu.h` itself, one layer down.

## C++ Performance Note

- **Target hardware/workload:** Chrome and Edge on desktop, discrete and
  integrated GPUs. No inference workload exists yet.

- **Performance budget:** None set. This packet deliberately does not
  establish latency or throughput budgets, because there is no workload to
  budget. Budgets are owed by BLLM-002, which selects the model.

- **Baseline measurement:** Not applicable — no steady-state path exists.
  What this packet does establish is the *measurement surface*: the page
  reports adapter description and the buffer limits that will constrain weight
  residency, so the first real baseline can record the device it was taken on.

- **Hot paths touched:** None. Device acquisition and pipeline creation are
  setup-time and run once.

- **Allocation/copy/serialization behavior:** One upload and one readback of a
  small buffer. Structurally, the design decision that matters is that
  `runtime/weights` will stream chunks to GPU buffers rather than holding a
  whole model in the WASM heap — WASM32's address space makes a resident
  400MB+ model wasteful and eventually impossible. That path is not built
  here, but `Buffer` is shaped to support chunked upload rather than a
  single `load_all`.

- **Concurrency model:** Single-threaded by constraint. No pthreads, because
  GitHub Pages cannot deliver COOP/COEP and therefore cannot provide
  `SharedArrayBuffer`. The harness runs in a plain Web Worker so the main
  thread stays responsive; this needs no shared memory since ownership passes
  by transfer.

- **Edge AI runtime/backend/configuration:** WebGPU via emdawnwebgpu. Compute
  only; no rendering pipeline is created.

- **Before/after measurement plan:** None applicable. Deferred to BLLM-002.

- **Accuracy/product-contract risk from performance optimizations:** None.
  No optimization is performed.

## Records

- Design discussion preceded this packet and settled: `webgpu.h` over
  JS-side WebGPU, single-threaded WASM, no bundler, weights fetched at runtime
  rather than committed, and Dawn deferred to the first model kernel.
- Deviation from the confirmed plan: doctest is pinned via CMake
  `FetchContent` at **v2.4.11 → v2.4.12** rather than vendored into
  `third_party/`. v2.4.11 declares a `cmake_minimum_required` below CMake 4's
  floor and will not configure at all. Pinning by tag is at least as
  deterministic as vendoring and avoids committing a third-party header.
- Toolchain deviation: Docker's registry access is broken on the development
  machine (the host reaches `auth.docker.io` in 0.2s; the daemon cannot reach
  it at all). The wasm build was produced with a host emsdk pinned to the same
  6.0.8 the image provides. CI continues to build in the container, and did so
  successfully, so the pinned environment is exercised on every push.
- Verified end to end on 2026-08-29: Chrome on Apple silicon (`apple` /
  `metal-3`), 4096 elements, 0 mismatches, deployed from CI to GitHub Pages.
- **Correction discovered during verification.** The design discussion treated
  `maxStorageBufferBindingSize` 128MB / `maxBufferSize` 256MB as hard caps.
  They are the spec's guaranteed minimums; this device reports 4096 MiB for
  both. Weight residency is therefore a portability decision rather than a
  fixed constraint. Recorded in
  `docs/research/2026-08-29-webgpu-limits-observed.md` and owed to BLLM-002.

# browser-llm

**Operating brand:** none. Internal R&D under Plainsight Systems LLC.

A self-built inference harness that runs an open-weight model entirely in the
browser. C++ compiled to WebAssembly, with compute executed on WebGPU. No
server-side inference and no remote model execution.

The point is to own the harness rather than adopt an existing one: to establish
what ordinary consumer hardware will actually do through a browser's GPU stack,
and to keep the execution path inspectable.

## Maturity

**Early. The build system and testable core exist; no inference does.**

As of 2026-08-29 this repository builds natively and runs unit tests. The
WebAssembly target, the WebGPU device path, and the web page are **not yet
written** — see [Current state](#current-state) for exactly what is and is not
present. No model is loaded and no weights are fetched by any code here.

## Repository structure

Directories marked *planned* do not exist yet. They are listed because the
layout is a decision, not an accident, and the reasoning matters more than the
current file count.

```
src/
  core/            platform-neutral C++. compiles unchanged in both targets.
    gpu/           webgpu.h device, buffers, pipelines, dispatch arithmetic
    gguf/          planned — container and metadata parsing
    quant/         planned — block layouts + CPU reference dequantization
    tokenizer/     planned
    model/         planned — architecture params from GGUF metadata
    kernels/       planned — dispatch logic for the WGSL below
    runtime/       planned — weight residency, KV cache, forward pass
    sampler/       planned
  shaders/         WGSL, embedded into the binary at build time
  wasm/            planned — the single Emscripten-aware translation unit
web/               planned — static page, worker, model cache. no npm.
tests/             native unit tests. no browser, no GPU.
tools/             boundary checks and helper scripts
cmake/             shader embedding codegen
docs/decisions/    project memory, packets, and inherited governance submodule
```

### Why this shape

**`src/core/` never knows it is in a browser.** It contains no `emscripten.h`,
no `EM_JS`, and no `#ifdef __EMSCRIPTEN__`. The same sources compile for the
native test binary and for WebAssembly. This is the one structural decision
that is expensive to reverse, because it determines whether inference logic can
be tested at all without launching a browser.

**The browser is the platform wrapper.** `src/wasm/` will be a single
translation unit exposing core to JavaScript, and `web/` owns the page. Product
behavior does not live there. Both invariants are enforced by
`tools/check_boundaries.sh` in CI rather than by review, because a reviewer
noticing a stray include is not a control.

**The GPU is reached through `webgpu.h`, not JavaScript.** In the browser,
`--use-port=emdawnwebgpu` implements that C API on top of the browser's WebGPU.
Natively, the same API is implemented by Dawn — the same engine Chrome itself
uses, with the same Tint shader compiler. So the same kernel code can run under
a native test binary and in the browser, and native tests exercise the real
stack rather than a mock. Driving WebGPU from JavaScript instead would put the
forward pass on the wrong side of the WASM boundary.

**No threads, by constraint.** GitHub Pages cannot set `Cross-Origin-Opener-Policy`
or `Cross-Origin-Embedder-Policy`, so `SharedArrayBuffer` is unavailable and
pthreads cannot be used. This is fine: the GPU does the compute, and WASM
parses, tokenizes, dispatches and samples. Responsiveness comes from running
the harness in a plain Web Worker, which needs no shared memory.

**Shaders are single-sourced.** WGSL files are compiled into the binary by
build-time codegen, and a unit test asserts the embedded constant is byte
identical to the file on disk. There is no second copy to drift.

**Weights are not in this repository, and cannot be.** GitHub caps files at
100MB and Pages does not serve Git LFS; a small quantized model is several
hundred megabytes. Weights will be fetched from a CDN on first run, verified by
SHA-256, and cached in OPFS. The *code* is self-contained; the weights are
fetched once. Saying otherwise would be a facade.

**No inference dependencies.** No llama.cpp, no ggml, no ONNX Runtime, no npm,
no bundler, no framework. Vendoring an existing inference stack would defeat
the premise. Third-party code is for things that are not the demonstration.

## Current state

| Component | State |
|---|---|
| Dual-target CMake build | present, native path verified |
| `core/gpu/dispatch_math` | present, unit tested |
| WGSL shader embedding | present, unit tested including drift detection |
| Native test suite (doctest) | present, 10 cases / 24 assertions passing |
| Boundary invariant checks | present, passing |
| CI: native build and test | present |
| `core/gpu` device, buffer, pipeline | **not written** |
| `src/wasm/bindings.cpp` | **not written** |
| `web/` page and worker | **not written** |
| CI: Pages deploy | present but `workflow_dispatch` only, because the wasm target does not build yet |
| Model loading, inference | **not started** — blocked on BLLM-002 |

Tracked in [`docs/decisions/packets/2026-08-29-repo-skeleton-and-build-system.md`](docs/decisions/packets/2026-08-29-repo-skeleton-and-build-system.md).

## Build

Native build and tests need only CMake and a C++20 compiler:

```sh
make test
```

Structural invariants:

```sh
make check
```

The WebAssembly build runs inside a pinned toolchain image, so local and CI use
the identical environment:

```sh
make wasm
```

It currently fails with an explicit message, because the wasm target has not
been written yet.

## Targets

Chrome and Edge on desktop are the blessed targets. Safari's WebGPU is
WebKit's own implementation rather than Dawn, so it is a separate verification
problem and is not assumed to follow.

The page must be served over http(s) — GitHub Pages, or `make serve` locally.
It cannot run from `file://`: module loading and the model fetch both fail
against a null origin.

## Open decisions

Target model, parameter count, quantization format, and performance budget are
undecided. They are BLLM-002 and nothing above depends on them.

## Governance

Entry point: [`docs/decisions/MEMORY.md`](docs/decisions/MEMORY.md).

Agents should read [`AGENTS.md`](AGENTS.md) first. This repo is C++-dominant and
performance-sensitive: both the C++ architecture gate and the C++ performance
gate bind non-trivial work. Inherited governance is a pinned submodule at
[`docs/decisions/governance/`](docs/decisions/governance/).

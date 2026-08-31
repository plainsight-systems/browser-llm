# Project Memory

This file is the canonical entry point for durable project context.

## Product Identity

- **Product:** browser-llm (working name)
- **Operating brand:** None. Internal R&D under the parent entity.
- **Parent entity:** Plainsight Systems LLC
- **Repository:** <https://github.com/plainsight-systems/browser-llm> (public).
  Deployed: <https://plainsight-systems.github.io/browser-llm/>

## Purpose

A self-built inference harness that runs an open-weight model entirely in the
browser: C++ compiled to WebAssembly, with compute executed on WebGPU. No
server-side inference, no remote model execution.

Status as of 2026-08-31: the toolchain reaches the GPU. The dual-target build,
platform-neutral core, WebGPU device path, wasm bindings and a static page all
exist, and CI deploys the page. No model is loaded and nothing is inferred.
BLLM-001 is **accepted**. BLLM-002 — model, quantization and the limits floor —
is next.

## Inherited Governance

Canonical list, public URLs, and internal/published status:
`inherited.md`.

- `plainsight_policies.md`
- `engineering_philosophies.md`
- `product_memory_workflow.md`
- `repo_creation_runbook.md`

## C++ Governance

This product is C++-dominant and performance-sensitive. All four C++ docs are
inherited and both gates bind:

- `cpp_architecture_playbook.md`
- `cpp_architecture_review.md`
- `cpp_performance_playbook.md`
- `cpp_performance_review.md`

## Locked Decisions

Decided 2026-08-31, on acceptance of BLLM-001:

- **The GPU is reached through the `webgpu.h` C API**, not from JavaScript, so
  the same code can later link native Dawn for deterministic kernel tests.
- **Single-threaded, no pthreads.** GitHub Pages cannot set COOP/COEP, so
  `SharedArrayBuffer` is unavailable. The harness runs in a plain Web Worker.
- **The device is asked for the adapter's advertised maxima** via
  `requiredLimits` rather than accepting WebGPU's defaults. Always
  satisfiable, so it costs no portability — but limits therefore vary per
  device and are known only after acquisition.
- **Async runs are serialised and generation-stamped** (`core/run_guard`).
  WebGPU cannot be cancelled, so a late callback must be identifiable as late
  rather than allowed to report a second result.

Decided 2026-08-28 during repo bootstrap:

- **Core implementation language is C++ compiled to WebAssembly via Emscripten.**
  Rationale: matches the edge-native/hot-path posture in
  `engineering_philosophies.md` and binds this repo to the C++ architecture and
  performance gates. Alternative considered: Rust via wasm-bindgen, rejected
  because it would leave the formal review gates unbound without authoring
  repo-local equivalents.
- **This repo is treated as C++-dominant and performance-sensitive.** Inference
  execution, model load, memory footprint, and GPU dispatch are
  performance-sensitive by default.
- **No operating brand.** Internal R&D attributed to Plainsight Systems LLC
  directly.
- **Version control is local-only for now.** `git init -b main`, no remote.
- **Inherited governance is a pinned submodule at
  `docs/decisions/governance/`.** `plainsight-systems-governance` was published
  (CC BY 4.0) with its internal content moved to the operations repo, so it can
  be checked out in-tree over HTTPS by anyone. Pinning is deliberate: it records
  which version of the review gates a packet was written and reviewed against,
  which a floating link cannot. The pin advances by explicit commit, not
  automatically; a stale pin is a record, not a defect. Deviation from
  `repo_creation_runbook.md`, which prescribes symlinks into a private sibling
  and should grow a public-repo branch.

## Open Questions

Not yet decided. These block the first implementation packet:

- Target model and parameter count.
- Quantization format and weight layout.
- Browser and GPU targets treated as blessed.
- Model delivery and caching strategy in the browser.
- Tokenizer ownership: built in-repo or vendored.
- WebGPU compute access path from WASM, and the boundary between C++ core and
  the JavaScript/TypeScript host.
- Performance budget and how baseline is measured.

## Research Index

- `research/2026-08-31-model-port-methodology.md` — method notes from prior
  in-house inference work: build a fixture ladder smallest-primitive-first,
  record provenance, and expect "reusable" ops to silently diverge from a
  model's reference. Our correctness gate cannot be byte-equal.
- `research/2026-08-31-gpu-readback-round-trip.md` — a serialized GPU round
  trip costs ~0.5 ms median. Read the sampled token back per step; do not build
  a speculative pipeline. Also carries the related lessons this measurement
  does not settle.
- `research/2026-08-29-webgpu-limits-observed.md` — observed WebGPU buffer
  limits are far above the spec's guaranteed minimums on capable hardware.
  Weight residency is a portability decision, not a fixed constraint.

## Active Workflow Pointers

- Queue: `QUEUE.md`
- Packets: `packets/`
- Workflow: `workflow.md`

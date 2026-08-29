# Project Memory

This file is the canonical entry point for durable project context.

## Product Identity

- **Product:** browser-llm (working name)
- **Operating brand:** None. Internal R&D under the parent entity.
- **Parent entity:** Plainsight Systems LLC
- **Repository:** local only at `repositories/tech-demos/browser-llm`. No remote
  created as of 2026-08-28.

## Purpose

A self-built inference harness that runs an open-weight model entirely in the
browser: C++ compiled to WebAssembly, with compute executed on WebGPU. No
server-side inference, no remote model execution.

Status: bootstrapped governance only. No implementation exists.

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
- **Inherited governance is referenced, not symlinked, in version control.**
  The runbook's symlink pattern assumes a private repo. This repo is intended to
  be public, and committed symlinks into a private sibling would dangle in every
  clone while advertising eight documents no outside reader can open. That is a
  documentation facade under `plainsight_policies.md`. Instead:
  `docs/decisions/inherited.md` is committed and states what is inherited, what
  is publicly readable, and what is internal; the local symlinks are gitignored
  and recreated by `scripts/link-governance.sh`. Deviation from
  `repo_creation_runbook.md`, which should grow a public-repo branch in its
  symlink guidance.
- **Scope posture: narrow slice at full quality, not demo-ware.** Per
  `engineering_philosophies.md`, "tech demo" here means the real harness on a
  deliberately narrow scope at full intended quality.

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

- None yet.

## Active Workflow Pointers

- Queue: `QUEUE.md`
- Packets: `packets/`
- Workflow: `workflow.md`

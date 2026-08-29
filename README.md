# browser-llm

**Operating brand:** none. Internal R&D under Plainsight Systems LLC.

## Purpose

A self-built inference harness that runs an open-weight model entirely in the
browser. C++ compiled to WebAssembly, with compute executed on WebGPU. No
server-side inference and no remote model execution.

The point is to own the harness rather than adopt an existing one: to establish
what ordinary consumer hardware will actually do through a browser's GPU stack,
and to keep the execution path inspectable.

## Maturity

**Planned. Nothing is implemented.**

As of 2026-08-28 this repository contains governance scaffolding only. There is
no source, no build, and no running demo. Everything below the Purpose heading
describes intent, not shipped behavior.

## Scope Posture

Per `engineering_philosophies.md`, this is not demo-ware. The intended
deliverable is the real harness on a deliberately narrow slice — a single model,
a single quantization, a named set of blessed browser and GPU targets — at full
intended quality. Narrowing the slice is the accepted way to cut work; lowering
the quality inside the slice is not.

## Open Decisions

The target model, quantization format, blessed browser and GPU targets, the
C++/JavaScript boundary, and the performance budget are all **undecided**. They
are tracked under Open Questions in `docs/decisions/MEMORY.md` and are the
subject of the first architecture packet.

## Build

Not yet defined. The core is intended to be C++ compiled via Emscripten; the
toolchain, build entry point, and supported host environments will be recorded
here once the first architecture packet settles them.

## Governance

Entry point: [`docs/decisions/MEMORY.md`](docs/decisions/MEMORY.md).

Agents should read [`AGENTS.md`](AGENTS.md) first. This repo is C++-dominant and
performance-sensitive: both the C++ architecture gate and the C++ performance
gate bind non-trivial work.

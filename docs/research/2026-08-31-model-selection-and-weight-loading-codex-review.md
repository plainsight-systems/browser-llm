OUTCOME: changes_requested

## Summary

BLLM-002 is not implemented. The repository contains no GGUF reader, Q4_0 implementation, tokenizer, model configuration, residency planner/uploader, fixtures, load entry point, or corresponding tests. All twelve acceptance criteria remain unchecked.

The repository is not presently facading success: the packet is marked `draft`, the queue says it awaits implementation, and the page explicitly says it does not load a model. However, this submission cannot pass an implementation review, and several packet contracts need correction before implementation.

The existing BLLM-001 native suite passes: 25 test cases and 74 assertions. Those tests do not cover BLLM-002.

## Findings

### [P1] The reviewed implementation and verification surface do not exist

Files: [CMakeLists.txt:21](/Users/andyhunter/repositories/tech-demos/browser-llm/CMakeLists.txt:21), [tests/CMakeLists.txt:12](/Users/andyhunter/repositories/tech-demos/browser-llm/tests/CMakeLists.txt:12), [worker.js:15](/Users/andyhunter/repositories/tech-demos/browser-llm/web/worker.js:15)

Why this matters: The core target contains only `dispatch_math.cpp`; the test binary contains only BLLM-001 tests; and the worker invokes only the self-check or readback benchmark. None of the interfaces promised at [packet:204](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-31-model-selection-and-weight-loading.md:204) exist. Therefore parsing safety, error paths, ownership, GPU cleanup, quantized upload, and every acceptance criterion are unreviewable.

Expected fix: Implement and wire the packet’s source, browser, fixture, and test surfaces, then resubmit with the acceptance boxes and manual evidence tied to actual artifacts.

### [P1] The proposed full-buffer parser contract conflicts with bounded streaming

File: [2026-08-31-model-selection-and-weight-loading.md:228](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-31-model-selection-and-weight-loading.md:228)

Why this matters: The packet says the parser borrows one `std::span<const std::byte>`, the caller owns “the bytes,” and bounds are checked against the actual buffer. It also guarantees that the complete 420 MB model never enters the WASM heap at [line 39](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-31-model-selection-and-weight-loading.md:39).

A span representing the whole GGUF requires the whole file to be resident. A span representing only a prefix cannot validate tensor regions later in the file against `span::size()`. The packet supplies no incremental source, bounded-window, or total-file-length contract that resolves this.

`R.2` supports `span` as a range-aware non-owning interface, but a span alone does not provide streaming or storage ownership. `SL.con.3` requires actual bounds safety, and `ES.103` also requires overflow-safe `offset + length` and shape-product arithmetic.

Expected fix: Define an explicit streaming architecture before coding—for example, an incremental metadata reader over bounded windows carrying the authoritative file length, followed by separately validated tensor regions. Tests should split valid and malformed fields across every chunk boundary and exercise checked addition/multiplication.

### [P1] A sampled block cannot prove the packet’s headline byte-identity contract

File: [2026-08-31-model-selection-and-weight-loading.md:149](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-31-model-selection-and-weight-loading.md:149)

Why this matters: The intent claims that streamed quantized weights are “proven byte-identical by readback” at [lines 8–11](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-31-model-selection-and-weight-loading.md:8), but acceptance reads back only one sampled block. That can pass when another chunk was truncated, copied to the wrong offset, assigned to the wrong buffer, or silently corrupted. It does not establish that the model is resident correctly.

`GPU.1` permits diagnostic readback but requires transfers and synchronization to be budgeted. `TLM.6` requires diagnostic/instrumented measurements to remain distinct from clean throughput baselines.

Expected fix: Either:

- Perform complete chunked readback verification in a clearly labeled diagnostic/manual mode, hashing or comparing every resident byte without materializing the whole model; or
- Narrow the stated contract to sampled verification and explicitly document its probabilistic coverage.

For the packet’s central risk claim, complete diagnostic verification is the defensible choice.

### [P1] Tokenizer self-roundtrip is not an independent correctness oracle

File: [2026-08-31-model-selection-and-weight-loading.md:145](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-31-model-selection-and-weight-loading.md:145)

Why this matters: `decode(encode(text)) == text` can pass while `encode()` produces entirely wrong token IDs. Multiple token sequences can decode to the same bytes, and an internally consistent but Qwen-incompatible tokenizer would silently feed the model incorrect inputs. This is precisely the kind of trivial assertion the review mandate asks reviewers to reject.

Expected fix: Commit independently generated fixtures that assert:

- Exact input bytes to exact token-ID sequences.
- Exact token-ID sequences to output bytes.
- Special-token recognition and allowed/disallowed policy.
- Multi-byte UTF-8 and byte-fallback behavior.
- Fixture provenance, reference implementation, version, and model snapshot.

Roundtrip may remain as an additional property, not the primary oracle.

### [P1] The residency planner is specified against an incomplete limit surface

Files: [device.h:11](/Users/andyhunter/repositories/tech-demos/browser-llm/src/core/gpu/device.h:11), [2026-08-31-model-selection-and-weight-loading.md:151](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-31-model-selection-and-weight-loading.md:151)

Why this matters: `DeviceLimits` claims four values constrain model layout, but omits at least `minStorageBufferOffsetAlignment` and `maxStorageBuffersPerShaderStage`. The packet’s planner acceptance criterion names only `maxStorageBufferBindingSize`, even though:

- Buffer creation is constrained by `maxBufferSize`.
- Bound ranges are constrained separately by `maxStorageBufferBindingSize`.
- Suballocated binding offsets must satisfy `minStorageBufferOffsetAlignment`.
- A tensor spanning buffers affects per-stage binding counts.

These are separate WebGPU constraints in the official [WebGPU limits specification](https://gpuweb.github.io/gpuweb/#limits). `GPU.2` also specifically requires suballocation alignment to preserve the intended lane-to-address layout. As written, the planner can produce a residency map that uploads successfully but cannot be bound by BLLM-003.

Expected fix: Make the planner consume every granted limit its output depends on. Add deterministic cases for allocation size, binding-window size, required alignment, spanning, and binding-count feasibility. Clearly distinguish physical buffers from bindable ranges.

### [P2] E.25 is cited for the wrong responsibility

File: [2026-08-31-model-selection-and-weight-loading.md:228](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-31-model-selection-and-weight-loading.md:228)

Why this matters: The packet says errors return as values “following E.25,” and repeats that claim at [line 303](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-31-model-selection-and-weight-loading.md:303). The consulted rule says:

- `E.25`: simulate RAII for resource management when exceptions are unavailable.
- `E.27`: use error codes systematically when exceptions are unavailable.
- `R.1`: encapsulate acquire/release pairs in automatic resource handles.

The proposed `UniqueHandle` use belongs under `E.25`/`R.1`; named parser/load result values belong under `E.27`. Conflating cleanup and error propagation leaves the actual error contract underspecified.

Expected fix: Correct the packet record and specify one systematic, `[[nodiscard]]` result/error representation for parsing, configuration, upload, cache, hash, and readback failures.

### [P1] No required model-load performance evidence exists

File: [2026-08-31-model-selection-and-weight-loading.md:256](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-31-model-selection-and-weight-loading.md:256)

Why this matters: The packet establishes a sub-100 ms parse budget and requires recorded upload throughput, chunk-size justification, and peak WASM heap on the real model. None exists. The current readback spike measures a different operation.

The performance checklist classifies a performance-sensitive change without target-hardware measurement as P1. `GPU.1` requires the bytes, measured transfer bandwidth, and synchronization component to be budgeted. `MEM.9` supports an explicit initialization boundary and a checked fixed-footprint steady state. `TLM.2` and `TLM.6` require memory instrumentation and clean throughput measurements to be separated and labeled.

Expected fix: Supply a real-model record naming device, browser build, granted limits, build profile, network versus OPFS source, timing boundaries, chunk size, upload count, bytes transferred, peak heap, and diagnostic overhead.

## C++ Architecture Review

Outcome: changes_requested

### Findings

- [P1] All proposed BLLM-002 components and deterministic tests are absent.
- [P1] The span-based reader and bounded streaming invariants are not architecturally compatible as specified.
- [P1] The planner’s limit interface cannot guarantee future bindability.
- [P1] The tokenizer and residency tests do not pin their stated product contracts.
- [P2] The exception-disabled error strategy cites the resource-management rule instead of the systematic-error rule.

### Boundary Assessment

- Core/wrapper separation: The existing BLLM-001 boundary check passes. There is no BLLM-002 boundary to assess.
- Public/private dependency classification: Not assessable; the proposed targets and headers do not exist.
- Header/API surface: Not assessable. The packet’s proposed reader and plan APIs need revision first.
- Ownership/lifetime clarity: GPU `UniqueHandle` direction is sound under `R.1` and `E.25`; parser storage lifetime and streaming ownership are unresolved.
- Test surface: Missing. The planned tokenizer and sampled-readback tests are also insufficient as contract oracles.

### Residual Risk

Any implementation started from the current packet risks either loading the complete file into WASM memory or weakening bounds checks, and may produce GPU storage that later kernels cannot bind.

## C++ Performance Review

Outcome: changes_requested

This gate applies: the work directly covers model load, peak memory, CPU-to-GPU transfer, and GPU residency.

### Findings

- [P1] No target-browser baseline, upload measurement, or real-model heap measurement exists.
- [P1] The central residency verification samples one block rather than verifying the transferred payload.
- [P1] The planner omits alignment and binding feasibility inputs that affect usable device layout.

### Performance Assessment

- Target hardware/workload: Chrome and Edge desktop, loading the real Qwen3-0.6B Q4_0 GGUF.
- Budget: Metadata parse under 100 ms; upload and SHA-256 presently unbudgeted but required to establish baselines.
- Baseline: Missing.
- Before/after result: Not applicable to optimization, but the required initial baseline is absent.
- Hot paths changed: None yet; no implementation exists.
- Allocation/copy behavior: Proposed only. Chunk size, copy count, staging behavior, and peak heap are unmeasured.
- Concurrency behavior: Proposed single-threaded worker; no load path exists to assess.
- Edge AI runtime configuration: No external runtime, as intended.

### Residual Risk

The proposed measurement instrumentation may perturb the numbers it records. Per `TLM.2` and `TLM.6`, heap diagnostics and clean transfer-throughput runs should be separate labeled artifacts.

## MCP grounding

Both required servers were available. No MCP call was cancelled, denied, or returned an error.

- `cpp-guidelines`
  - Called `search_guidelines` for untrusted binary parsing/bounds/overflow, exception-disabled error handling, borrowed spans/lifetimes, RAII handles, and ignored error results.
  - Called `get_guideline` for `SL.con.3`, `ES.103`, `R.2`, `E.25`, `E.27`, `R.1`, `P.11`, and `SL.4`.
- `cpp-performance`
  - Called `search_guidelines` for bounded model loading, peak-memory instrumentation, GPU transfer batching, device layout/alignment, initialization allocation, and baseline methodology.
  - Called `get_guideline` for `MEM.7`, `MEM.9`, `GPU.1`, `GPU.2`, `GPU.9`, `TLM.2`, and `TLM.6`.

No REVIEW ENVIRONMENT FAILURE occurred for either MCP server.

## Residual risk

- No GGUF, tokenizer, config, upload, OPFS, SHA-256, or real-browser behavior could be executed because it does not exist.
- The existing 25 native tests passed, but they cover only prior work.
- The static boundary and diagram checks passed.
- The mutation-based boundary-guard test was NOT PERFORMED because this review environment is read-only and the test attempted to create `src/core/gpu/_boundary_probe.h`.
- External model contents and configuration provenance were not independently revalidated; the repository currently contains no pinned model artifact, hash, or fixture against which to do so.
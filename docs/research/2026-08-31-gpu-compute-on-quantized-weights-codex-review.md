OUTCOME: changes_requested

## Summary

BLLM-003 is not implemented. The repository accurately labels it `draft`, leaves every acceptance criterion unchecked, and explicitly documents native Dawn tests as absent. This is not a no-facades violation, but there is no implementation to approve.

The packet also contains three blocking design defects: its numerical bound is not a justified bound for two independent floating-point evaluations, the proposed dispatch interface does not define ownership or failure behavior, and the corruption test can fail to demonstrate that the oracle has teeth.

Existing verification passed: boundary checks, diagram checks, and 25 existing native tests/74 assertions. None exercise BLLM-003.

## Findings

### [P1] No packet implementation or acceptance evidence exists

File: [CMakeLists.txt:21](/Users/andyhunter/repositories/tech-demos/browser-llm/CMakeLists.txt:21), [tests/CMakeLists.txt:12](/Users/andyhunter/repositories/tech-demos/browser-llm/tests/CMakeLists.txt:12), [.github/workflows/test.yml:8](/Users/andyhunter/repositories/tech-demos/browser-llm/.github/workflows/test.yml:8)

Why it matters: `browser_llm_core` still contains only `dispatch_math.cpp`; GPU sources remain wasm-only. The only embedded shader is `vector_add.wgsl`. There is no Dawn or SwiftShader dependency, fused kernel, Q4_0 implementation, CPU matmul, named GPU test, CI cache, lane mapping, browser/native comparison, or baseline record. The README confirms native Dawn tests are absent at [README.md:119](/Users/andyhunter/repositories/tech-demos/browser-llm/README.md:119).

Expected fix: Complete BLLM-002 first, implement BLLM-003, record every acceptance artifact, and request a new implementation review. Until then, retain `draft`; do not mark the packet review-approved or accepted.

### [P1] The proposed tolerance is not a valid hard comparison bound

File: [2026-08-31-gpu-compute-on-quantized-weights.md:108](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-31-gpu-compute-on-quantized-weights.md:108)

Why it matters: `K·ε·Σ|aᵢbᵢ|` is presented as an approximate error estimate and then promoted into a hard “anything above is a defect” postcondition. The test compares two independently rounded computations, so the bound must cover both CPU and WGSL multiplication/accumulation error, not merely one accumulation. WGSL also permits unspecified rounding direction, reassociation, fusion, and flushing eligible subnormal inputs/results. The [WGSL floating-point specification](https://gpuweb.github.io/gpuweb/wgsl/#floating-point-evaluation) expressly permits reassociation and fusion and does not specify a rounding mode. An approximate one-sided estimate is therefore insufficient as the postcondition required by I.7.

Expected fix: Derive a comparison bound from the exact CPU and WGSL algorithms, using separate bounds for both results—normally γ-style bounds including products and additions—then sum those bounds. Define handling of cancellation, overflow, subnormals, NaN/infinity, FMA, and test-input ranges. Keep the bound computed from actual operands.

### [P1] The dispatch API cannot support its claimed ownership, error, and allocation contracts

File: [2026-08-31-gpu-compute-on-quantized-weights.md:192](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-31-gpu-compute-on-quantized-weights.md:192), [same packet:205](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-31-gpu-compute-on-quantized-weights.md:205), [same packet:253](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-31-gpu-compute-on-quantized-weights.md:253)

Why it matters: The proposed entry takes weights, activation, and dimensions, but specifies neither an output destination nor a result/error type. It does not say who owns pipelines, bind groups, output buffers, command objects, or test readback buffers. Consequently, “no per-dispatch allocation” is unsupported: the API either allocates output/resources internally or depends on undocumented ambient state. It also omits preconditions such as Q4 block divisibility, byte ranges, alignment, compatible dimensions, and device limits.

With exceptions disabled, construction and dispatch failures require systematic, observable error values under E.25 and E.27. Preconditions and postconditions belong in the interface under I.5 and I.7. WebGPU resources should remain automatically released under R.1.

Expected fix: Define a cohesive setup/dispatch contract. A likely shape is a move-only kernel object owning the pipeline and reusable bindings, with dispatch borrowing explicit weight, activation, output, and workspace buffers and returning a `[[nodiscard]]` typed status. Define all range/alignment/dimension preconditions and capture WebGPU validation failures observably.

### [P1] The corruption test does not define a mutation that must fail the gate

File: [2026-08-31-gpu-compute-on-quantized-weights.md:108](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-31-gpu-compute-on-quantized-weights.md:108), [same packet:139](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-31-gpu-compute-on-quantized-weights.md:139)

Why it matters: The packet says CPU and GPU consume the same Q4_0 weights. If the shared source block is corrupted before both computations, the error remains common-mode and the comparison passes. Even GPU-only corruption can remain below tolerance when its activation is zero or its contribution is small. Thus the criterion can be implemented without proving that the intended gate detects a meaningful defect, contrary to I.7.

Expected fix: Compute and retain the CPU oracle from pristine bytes, mutate only the uploaded GPU copy, and choose deterministic scale, nibble, and activation values whose analytically predicted output delta exceeds the derived bound. Record the negative-control run separately from the permanently green test suite.

### [P2] Dawn is described at the wrong dependency boundary

File: [2026-08-31-gpu-compute-on-quantized-weights.md:196](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-31-gpu-compute-on-quantized-weights.md:196), [CMakeLists.txt:35](/Users/andyhunter/repositories/tech-demos/browser-llm/CMakeLists.txt:35)

Why it matters: The packet says a native test target links Dawn, while `browser_llm_core` is supposed to compile `device.cpp` and `self_check.cpp` natively and exposes headers that include `webgpu.h`. Linking Dawn only at the test executable risks undeclared include/link dependencies and reliance on ambient or transitive configuration.

Expected fix: Specify whether Dawn is part of `browser_llm_core`’s public or private CMake interface. Because public core headers expose WebGPU types, the native header dependency likely must be public; static-link requirements must also propagate correctly. Tests should normally link the core target rather than reconstruct its implementation dependency graph.

### [P2] The claimed GPU.1 readback budget is incomplete and outside this packet’s scope

File: [2026-08-31-gpu-compute-on-quantized-weights.md:237](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-31-gpu-compute-on-quantized-weights.md:237), [same packet:58](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-31-gpu-compute-on-quantized-weights.md:58)

Why it matters: The packet says it records bytes, measured transfer, measured wait, and delayed steps, but supplies only combined round-trip latency and a projected percentage. It omits bytes, separate transfer/wait measurements, queue/fence waited on, and delayed steps. The underlying research also says behavior under real forward-pass load remains unmeasured. GPU.1 requires the hot-loop readback boundary and measured cost to be explicit.

Expected fix: Either remove this future decode-loop budget from BLLM-003 or move a complete budget into the decode packet, including bytes, staging path, queue boundary, transfer and wait components, delayed steps, target/browser, and under-load measurement.

## C++ Architecture Review

Outcome: changes_requested

### Findings

- P1: No BLLM-003 implementation or independent test surface exists.
- P1: Dispatch ownership, output lifetime, preconditions, postconditions, and error propagation are undefined.
- P1: The negative-control test does not guarantee an observable defect.
- P2: Dawn dependency visibility is not assigned to the component that consumes and exposes it.

### Boundary Assessment

- Core/wrapper separation: Proposed direction is sound; current `tools/check_boundaries.sh` passes.
- Public/private dependency classification: Incomplete for native Dawn.
- Header/API surface: The proposed dispatch API is underspecified.
- Ownership/lifetime clarity: Existing `UniqueHandle` is sound under R.1/E.25, but ownership of new kernel resources is not designed.
- Test surface: Planned independent GPU tests are appropriate, but absent and partly underspecified.

### Residual Risk

- Native Dawn may expose callback/event-processing assumptions currently compiled only under emdawnwebgpu.
- BLLM-003 depends on BLLM-002 types, layout, and residency behavior that do not yet exist.

## C++ Performance Review

Outcome: changes_requested

This gate applies: fused inference matmul, GPU dispatch, weight layout, and allocation behavior are performance-sensitive by repository policy.

### Findings

- P1: No baseline or target-hardware kernel measurement exists.
- P1: “No per-dispatch allocation” has no supporting ownership/API design; GPU.9 requires steady-state allocation to remain outside the hot loop.
- P2: The GPU.1 readback budget is incomplete.
- The lane-address acceptance criterion is well-shaped and follows GPU.2.
- Deferring tiling/shared memory until reuse is measured is sound under GPU.5.
- Creating a baseline before optimization is sound under GPU.10, but the future baseline must separate CPU submission, queue wait, GPU execution, and readback.

### Performance Assessment

- Target hardware/workload: Claimed Chrome/Edge desktop, native Metal, and SwiftShader; not exercised.
- Budget: Explicitly none.
- Baseline: None.
- Before/after result: None; no kernel exists.
- Hot paths changed: None yet.
- Allocation/copy behavior: Claimed, not designed or measured.
- Concurrency behavior: Single-threaded host remains unchanged.
- Edge AI runtime configuration: Dawn is planned as the native test implementation, not an inference runtime.

## MCP grounding

Both required servers were available. No REVIEW ENVIRONMENT FAILURE occurred.

- `cpp-guidelines.search_guidelines` was called for exception-disabled error handling, RAII, bounds/overflow, interface contracts, and floating-point comparisons.
- `cpp-guidelines.get_guideline` was called for I.5, I.7, E.25, E.27, R.1, SL.con.3, and ES.103.
- `cpp-performance.search_guidelines` was called for GPU lane layout, transfers/readback, shared memory, allocation, batching, and measurement.
- `cpp-performance.get_guideline` was called for GPU.1, GPU.2, GPU.5, GPU.6, GPU.7, GPU.9, and GPU.10.
- The current official WGSL specification was additionally consulted for floating-point accuracy, rounding, reassociation, fusion, and flush-to-zero behavior.

## Residual risk

Because no BLLM-003 code exists, correctness, resource cleanup on every GPU failure path, browser/native shader identity, real-shape dispatch, SwiftShader operation, build-cache behavior, and performance measurements remain unreviewed. The present green tests establish only the previously accepted BLLM-001 surface.
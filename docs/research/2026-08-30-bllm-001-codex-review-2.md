OUTCOME: changes_requested

## Summary

BLLM-001 is substantially real: the current boundary check passes, the existing native test binary passes 17 cases/48 assertions, shader generation is byte-identical, WebGPU handles use tested RAII wrappers, and WASM artifacts exist.

Acceptance is nevertheless blocked. The acquired device’s effective limits are not the limits exposed by `Device::limits()`. The code reports adapter maxima while requesting a default-limits device, undermining the packet’s dispatch and future weight-residency contract. Several claimed fixes also remain incomplete.

## Findings

### [P1] Adapter maxima are misrepresented as the acquired device’s usable limits

File: [src/core/gpu/device.cpp:105](/Users/andyhunter/repositories/tech-demos/browser-llm/src/core/gpu/device.cpp:105)

The code queries `wgpuAdapterGetLimits`, stores the result in `Device::limits()`, then requests a device without `requiredLimits` at [device.cpp:116](/Users/andyhunter/repositories/tech-demos/browser-llm/src/core/gpu/device.cpp:116). `run_self_check` subsequently treats those adapter values as the device’s dispatch limits at [self_check.cpp:46](/Users/andyhunter/repositories/tech-demos/browser-llm/src/core/gpu/self_check.cpp:46).

WebGPU creates a default-limits device unless better limits are explicitly requested, and validation uses the device’s exact limits—not the adapter’s maxima. Thus an adapter reporting 4 GiB buffers can yield a device limited to the default 256/128 MiB values. [The WebGPU specification states this explicitly.](https://gpuweb.github.io/gpuweb/#limits)

Why this matters: `Device::limits()` has a materially false contract, contrary to explicit-interface guidance (I.1). The page and durable research present unsupported capacity as the measurement surface for BLLM-002.

Expected fix: query `wgpuDeviceGetLimits` after acquisition and use those values for dispatch and reporting. Report adapter maxima separately if useful. When BLLM-002 chooses a floor, request only the required better limits through `requiredLimits` and fail acquisition if unavailable. Correct the packet and research conclusions accordingly.

### [P2] Adapter-info query failure is still rendered as ordinary missing metadata

Files: [src/core/gpu/device.cpp:90](/Users/andyhunter/repositories/tech-demos/browser-llm/src/core/gpu/device.cpp:90), [src/wasm/bindings.cpp:94](/Users/andyhunter/repositories/tech-demos/browser-llm/src/wasm/bindings.cpp:94), [web/app.js:55](/Users/andyhunter/repositories/tech-demos/browser-llm/web/app.js:55)

The fix adds and serializes `adapter.queried`, but `app.js` never reads it. A failed `wgpuAdapterGetInfo` still produces the same “(not reported)” presentation as a successful query with legitimately empty fields.

Why this matters: the previous review finding was only partially fixed, while the packet records it as complete. That is a no-facades defect. It also leaves error handling non-systematic (E.27) and the observable interface ambiguous (I.1).

Expected fix: either fail acquisition when adapter metadata is contractually required, or render an explicit “adapter information query failed” state. Add a deterministic result-rendering test for both query failure and legitimately empty fields.

### [P2] Async callback ownership and preconditions are not represented safely

Files: [src/core/gpu/device.h:45](/Users/andyhunter/repositories/tech-demos/browser-llm/src/core/gpu/device.h:45), [src/core/gpu/device.cpp:52](/Users/andyhunter/repositories/tech-demos/browser-llm/src/core/gpu/device.cpp:52), [src/core/gpu/self_check.h:24](/Users/andyhunter/repositories/tech-demos/browser-llm/src/core/gpu/self_check.h:24), [src/core/gpu/self_check.cpp:24](/Users/andyhunter/repositories/tech-demos/browser-llm/src/core/gpu/self_check.cpp:24)

Both public callbacks must be non-null, but that precondition is neither stated nor checked; each is invoked unconditionally. A null callback therefore causes invalid pointer use (I.5, I.12, ES.65). `RequestCallback` also receives an error pointer into a temporary string whose callback-only lifetime is undocumented.

The bridging contexts use naked `new`, `delete self`, and `delete this`. That is manual ownership outside the otherwise sound handle RAII system and is not covered by the `UniqueHandle` tests (R.11, ES.24). Consequently, the packet’s broad “RAII throughout” claim is overstated, although the WebGPU handle portion does correctly follow R.1, C.31, and E.25.

Expected fix: define named result objects with owned error values and explicit callback-lifetime documentation; express or enforce the non-null callback precondition. Reacquire callback contexts into `std::unique_ptr` at callback entry or otherwise isolate manual ownership in a reviewed resource manager. Add fake-async tests covering every completion path.

### [P2] Construct-on-first-use did not eliminate hidden mutable singleton state

File: [src/wasm/bindings.cpp:64](/Users/andyhunter/repositories/tech-demos/browser-llm/src/wasm/bindings.cpp:64)

`active_device()` and `request_in_flight()` remain mutable function-local statics controlling all calls. Moving them behind accessors solves static initialization ordering, but it does not make dependencies explicit or stop them being singleton state (I.1, I.2, R.6).

The performance corpus’s LIFE.6 specifically describes construct-on-first-use as an initialization-order technique and warns that it is not a singleton endorsement. The packet therefore incorrectly records the I.2/R.6 issue as fixed by LIFE.6.

Expected fix: represent module/run state as an explicit object and thread it through callback userdata. If module-wide ambient state is unavoidable at the C export boundary, document the exception and test lifecycle, re-entry, and teardown behavior rather than claiming the guideline concern is gone.

### [P2] The “exact” Emscripten allowlist is regex-inexact

File: [tools/check_boundaries.sh:39](/Users/andyhunter/repositories/tech-demos/browser-llm/tools/check_boundaries.sh:39)

The checker excludes the permitted path using:

```sh
grep -v "^${WRAPPER}$"
```

Because `.` is a regular-expression wildcard, a second file such as `src/wasm/bindingsXcpp` is also excluded. The regression test at [test_check_boundaries.sh:45](/Users/andyhunter/repositories/tech-demos/browser-llm/tests/test_check_boundaries.sh:45) uses an unrelated filename and does not catch this near-collision.

Why this matters: the current source tree satisfies the boundary, but the control does not enforce the exact allowlist it claims to enforce.

Expected fix: use fixed-string exact matching, such as `grep -Fvx "${WRAPPER}"`, and add a near-collision fixture.

### [P2] The doctest dependency is not deterministically pinned

File: [tests/CMakeLists.txt:5](/Users/andyhunter/repositories/tech-demos/browser-llm/tests/CMakeLists.txt:5)

`GIT_TAG v2.4.12` is a mutable tag, and `GIT_SHALLOW TRUE` forces fresh configurations to resolve it remotely. This contradicts the packet’s statement that a tag is “at least as deterministic as vendoring.”

CMake’s own [FetchContent documentation](https://cmake.org/cmake/help/latest/module/FetchContent.html) recommends a commit hash because branch and tag names do not preserve traceability and repeatability.

Expected fix: remove `GIT_SHALLOW TRUE` and pin the currently resolved commit, `1da23a3e8119ec5cce4f9388e91b065e20bf06f5`, or use a hash-verified archive/vendor strategy. Document that a clean configure requires network access unless an offline dependency path is provided.

### [P3] The verification table omits the ownership test supporting the packet

File: [packet:77](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-29-repo-skeleton-and-build-system.md:77)

The table claims to enumerate native logic tests but lists only dispatch arithmetic and shader embedding. The architecture note and review record rely materially on `unique_handle_test.cpp`.

Expected fix: add `unique_handle_test` and the ownership cases it pins to the verification table.

## C++ Architecture Review

Outcome: changes_requested

### Findings

- [P1] `Device::limits()` exposes adapter maxima rather than effective device limits.
- [P2] Adapter metadata failure remains hidden at the UI boundary.
- [P2] Async callback contracts and context ownership are insufficiently explicit.
- [P2] Wrapper lifecycle remains hidden singleton state.
- [P2] The exact boundary allowlist is not exact.
- [P2] The test dependency’s claimed reproducibility does not hold.

### Boundary Assessment

- Core/wrapper separation: The current source tree is clean, and the wrapper remains mostly translational. The automated allowlist has a bypass.
- Public/private dependency classification: Direction is currently inward. Native compilation of `device.cpp` and `self_check.cpp` remains explicitly deferred rather than falsely claimed.
- Header/API surface: Headers are self-contained, but `Device::limits()` has the wrong semantics and callback lifetime/error contracts are unclear.
- Ownership/lifetime clarity: WebGPU handles are soundly wrapped and tested. Async CPU-side contexts and module state remain manually or implicitly owned.
- Test surface: Dispatch arithmetic, shader embedding, and `UniqueHandle` pass native tests. Device acquisition, effective-limit behavior, metadata failure, device loss, and GPU execution have no deterministic automated test.

### Residual Risk

- Native Dawn integration may reveal callback and initialization assumptions currently compiled only under emdawnwebgpu.
- There is no automated browser test pinning the page’s success and failure contracts.

## C++ Performance Review

Not applicable: this packet’s GPU dispatch is a one-shot 4,096-element toolchain self-check with setup-time device/pipeline creation, not inference, model loading, sustained memory processing, or a steady-state hot path.

## MCP grounding

- `cpp-guidelines/search_guidelines` was called for RAII and exception-disabled cleanup, globals/singletons, callback preconditions and lifetime, raw async ownership, initialization, interfaces, and systematic error handling.
- `cpp-guidelines/get_guideline` was called for I.1, I.2, R.6, I.5, I.6, I.12, ES.65, R.11, ES.24, R.1, C.31, E.25, E.27, E.4, I.11, and R.20.
- `cpp-performance/search_guidelines` was called for setup-time applicability, GPU capability measurement, and static/resource lifetime.
- `cpp-performance/get_guideline` was called for LIFE.6, MEM.9, and COPY.4.
- **REVIEW ENVIRONMENT FAILURE: none.** All required MCP searches and lookups completed successfully.

## Residual risk

The existing native executable passed 17/17 cases and 48/48 assertions; the current boundary check passed; and the generated shader matched its WGSL source byte-for-byte. A clean configure/build was not performed because the workspace is read-only. Current GitHub workflow status and the deployed page could not be independently reverified because GitHub access failed.
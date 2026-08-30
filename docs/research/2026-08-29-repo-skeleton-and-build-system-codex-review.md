OUTCOME: changes_requested

## Summary

BLLM-001 is not ready for acceptance. The current browser path, shader embedding, ownership wrappers, and native logic tests are substantially real, but the packet’s central dual-target guarantee does not hold: native `browser_llm_core` compiles only `dispatch_math.cpp`, while the WebGPU device and self-check sources compile only under Emscripten.

A separate P1 defect in unpacketized review tooling can also allow an unreachable MCP server through the preflight that claims to prevent exactly that failure.

Local evidence:

- Boundary check passed against the current tree.
- Existing native test executable passed 17 cases and 48 assertions.
- Existing `.mjs` and `.wasm` artifacts are present.
- Normal `ctest` could not run because this read-only sandbox prevented CTest from writing `LastTest.log`.
- CI and deployed-page claims could not be independently reverified because outbound GitHub access was unavailable.

## Findings

### [P1] The native target does not compile the same core sources as the WASM target

File: [CMakeLists.txt:21](/Users/andyhunter/repositories/tech-demos/browser-llm/CMakeLists.txt:21)

`browser_llm_core` initially contains only `dispatch_math.cpp`; `device.cpp` and `self_check.cpp` are added only inside `if(EMSCRIPTEN)` at line 35. The generated native build rules confirm that only `dispatch_math.cpp` is compiled, while all three sources are compiled for WASM.

This directly contradicts the packet’s guaranteed invariant at [packet:35](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-29-repo-skeleton-and-build-system.md:35) and the README’s claim at [README.md:52](/Users/andyhunter/repositories/tech-demos/browser-llm/README.md:52). It also means the native target does not compile-check the main new core API or provide a usable native include dependency for `device.h`.

Expected fix: make native and WASM core targets compile the same core sources, supplying the native `webgpu.h` header dependency while deferring Dawn linkage if appropriate. Otherwise revise the architecture through an explicit decision and stop claiming source parity.

### [P1] MCP preflight passes an unreachable server and was added outside the packet

File: [scripts/codex-review.sh:63](/Users/andyhunter/repositories/tech-demos/browser-llm/scripts/codex-review.sh:63)

On connection failure, `curl -w '%{http_code}'` emits `000`, then `|| echo 000` appends another `000`. The resulting value is `000000`, so the comparison with `000` fails and the supposedly mandatory preflight passes. This was reproduced against an unused localhost port.

That invalidates the script’s stated guarantee that an unavailable guideline server stops review before the model call. The script was also added in two commits after the BLLM-001 implementation without appearing in BLLM-001 or another packet, contrary to the one-packet/one-intent workflow.

Expected fix: branch on `curl`’s exit status directly and add a deterministic unreachable-endpoint test. Move this review automation into its own packet and review it independently.

### [P2] Adapter metadata and limit-query failures are silently converted into default data

File: [src/core/gpu/device.cpp:88](/Users/andyhunter/repositories/tech-demos/browser-llm/src/core/gpu/device.cpp:88)

Failures from `wgpuAdapterGetInfo` and `wgpuAdapterGetLimits` are ignored. An information failure can therefore produce a successful page filled with “not reported” fields, while a limits failure becomes zeros and later produces the misleading error that the element count exceeds the device dispatch limit.

This violates the packet’s observable reporting contract and conflates “the adapter legitimately omitted a field” with “the metadata query failed.”

Expected fix: propagate each failed status explicitly, or represent query failure separately from legitimately empty fields. Add deterministic tests around result translation.

### [P2] The boundary check does not enforce the invariants it claims to control

File: [tools/check_boundaries.sh:17](/Users/andyhunter/repositories/tech-demos/browser-llm/tools/check_boundaries.sh:17)

The script passes the current tree, but its checks are weaker than the packet:

- It searches selected tokens such as `EMSCRIPTEN_KEEPALIVE`, not the promised general `EMSCRIPTEN_` pattern.
- It permits one arbitrary non-core source file to include `emscripten.h`; it does not require that file to be `src/wasm/bindings.cpp`.
- Its outward-dependency pattern detects only a narrow quoted include form, not the broader dependency invariant its comments claim to enforce.

Expected fix: use an exact allowlist for Emscripten-aware files, match the packet’s complete forbidden-token set, and add fixtures proving violations cause failure.

### [P2] Canonical workflow state describes a repository that no longer exists

Files: [MEMORY.md:10](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/MEMORY.md:10), [QUEUE.md:7](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/QUEUE.md:7)

`MEMORY.md` says there is no remote and no implementation. `QUEUE.md` says the device path, bindings, page, and Pages workflow remain unimplemented and blocked. All of those statements conflict with the current repository and the `review_requested` packet.

These files are the canonical session entry points, so the stale state misdirects future implementers and reviewers.

Expected fix: update both files to reflect the current implementation, remote, packet status, and genuine remaining review work.

### [P3] Test documentation no longer matches the test suite

Files: [packet:68](/Users/andyhunter/repositories/tech-demos/browser-llm/docs/decisions/packets/2026-08-29-repo-skeleton-and-build-system.md:68), [README.md:102](/Users/andyhunter/repositories/tech-demos/browser-llm/README.md:102)

The verification table lists only dispatch and shader tests even though the architecture note relies on `UniqueHandle` tests. The README says 10 cases and 24 assertions; the current executable reports 17 and 48.

Expected fix: update the packet’s verification table and avoid hard-coded test counts in the README unless they are maintained automatically.

## C++ Architecture Review

Outcome: changes_requested

### Findings

- [P1] The native and WASM core targets compile different source sets.
- [P2] Adapter information and limits failures have unclear, misleading error behavior.
- [P2] The automated boundary control does not fully enforce its documented invariant.

### Boundary Assessment

- Core/wrapper separation: The current source tree is clean; no core file references Emscripten or wrapper code, and the browser wrapper remains thin.
- Public/private dependency classification: Incomplete for native builds. The public device headers depend on `webgpu.h`, but the native target supplies neither that dependency nor the corresponding core sources.
- Header/API surface: Cohesive overall, but the native configuration cannot consume the full advertised core surface.
- Ownership/lifetime clarity: `UniqueHandle` exists, is move-only, and its deterministic tests pass. No manual `wgpuXRelease` calls remain outside the aliases. Guideline compliance was not established because full MCP lookups were denied.
- Test surface: Dispatch arithmetic, shader embedding, and generic handle ownership are tested natively. Device acquisition and GPU execution remain browser-only, and most new core sources are not even compiled natively.

### Residual Risk

Native Dawn integration may expose portability or lifetime assumptions currently hidden by compiling those sources only with emdawnwebgpu.

## C++ Performance Review

Not applicable: this packet implements a one-shot toolchain self-check and setup-time device/pipeline creation, with no inference, model-load, sustained memory, or steady-state hot path.

## MCP Grounding

**REVIEW ENVIRONMENT FAILURE**

- `cpp-guidelines/search_guidelines` succeeded for searches covering resource ownership and RAII, exception-disabled error handling, callback/precondition contracts, dependency isolation, and integer overflow.
- `cpp-guidelines/get_guideline` was denied with: “MCP tool call requires approval, but approval policy is never.”
- Result: **C++ Core Guidelines compliance check NOT PERFORMED.** No rule summary or packet citation was treated as authoritative in this review.
- `cpp-perf-guidelines/search_guidelines` succeeded for setup-time applicability, GPU upload/readback, and asynchronous lifetime topics.
- `cpp-perf-guidelines/get_guideline` was denied with the same approval-policy error.
- Result: **performance-corpus guideline check NOT PERFORMED.** The performance gate’s non-applicability above comes from the repository’s authoritative checklist and the actual execution scope, not an unverified MCP rule.

## Residual Risk

The review could not:

- perform the required full guideline audit;
- rebuild from a clean tree in the read-only sandbox;
- run standard CTest logging;
- independently validate the deployed page or current GitHub workflow status.

Acceptance requires fixing the P1 findings and rerunning the guideline audit in an environment where both `get_guideline` tools are permitted.
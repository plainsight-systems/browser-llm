# BLLM-003: GPU compute on quantized weights, gated in CI

**Status:** draft
**Change class:** architectural (includes a cross-cutting dependency change)

## Intent

- **What is changing:** The harness gains the ability to compute on quantized
  weights, and a deterministic gate proving it. Native Dawn is linked so the
  same WGSL runs in a native test binary — Metal locally, SwiftShader in CI —
  and the first kernel is a **fused dequantize-matmul**: read Q4_0 blocks,
  unpack in-register, accumulate. Verified against a CPU reference at the
  matrix shapes the model actually uses.

- **Why the change is necessary:** After BLLM-002 the weights are resident on
  the GPU and provably intact, but nothing can compute on them, and no kernel
  in this repo has a deterministic correctness gate. Both halves are required
  for either to be worth having: Dawn without a kernel proves nothing, a kernel
  without Dawn has no gate, and dequantization without matmul is not a
  capability — the production path never materialises dequantized weights,
  it unpacks them inside the multiply.

- **Expected behavior changes:** `make test` gains GPU kernel tests that run
  without a browser. CI gains the same on every push. The page gains nothing
  user-visible; this packet produces a verified capability, not a feature.
  Still no inference and no token generated.

- **Guaranteed invariants/contracts:**
  - **Dequantization happens inside the kernel.** No path materialises a
    dequantized weight buffer, on CPU or GPU. That would restore the memory
    cost quantization exists to avoid.
  - The same WGSL source runs natively and in the browser. No `#ifdef`, no
    second copy — the shader-embedding invariant from BLLM-001 extends here.
  - Numerical tolerance is **derived from the arithmetic**, stated in the test,
    and never widened to make a test pass. A failure above the bound is a bug,
    not noise.
  - Kernels are written for the shapes this model needs. No generic matmul
    abstraction — there is one model to satisfy.

## Scope

**In:** Dawn as a native dependency (Metal local, SwiftShader CI) with build
caching; the fused dequant-matmul kernel; a CPU reference; tolerance derivation;
tests at the model's real shapes; CI wiring.

**Explicitly out:**

- **Optimization.** No tiling, no workgroup tuning, no shared-memory staging.
  The kernel is naive and correct. This is not deferral-because-hard: the
  performance gate requires a baseline before optimizing, and this packet
  creates it. Optimizing here would mean tuning against a number that does not
  yet exist.
- Attention, RoPE, RMSNorm, softmax, sampling — later kernels on this harness.
- Any decode loop or KV cache.

## The dependency change (§6.5)

Dawn is a large dependency and this is the packet's main risk surface.

- **What changes:** native builds gain Dawn, built standalone via CMake with
  `-DDAWN_FETCH_DEPENDENCIES=ON -DDAWN_ENABLE_INSTALL=ON
  -DDAWN_BUILD_SAMPLES=OFF`. No depot_tools. CI additionally builds SwiftShader
  and points Dawn at it via `VK_ICD_FILENAMES`. The **wasm build is
  unaffected** — it continues to use `--use-port=emdawnwebgpu` and gains no
  dependency.
- **Why necessary:** it is the only way to execute our WGSL deterministically
  without a browser, and therefore the only way kernel correctness becomes a CI
  gate rather than a manual check.
- **Why Dawn and not wgpu-native:** `webgpu.h` is a multi-vendor standard, and
  wgpu-native ships prebuilt binaries that would be far cheaper. But
  wgpu-native does not yet implement the stable header and discrepancies with
  Dawn remain. We already compile against Dawn's flavor through emdawnwebgpu;
  introducing a second implementation would add divergence in precisely the
  layer being validated. Cost accepted deliberately.
- **Risk surface:** build time and CI cache behavior (unknown until measured —
  an early task, not an end-of-packet discovery); Dawn version pinning, by
  commit not tag; and the numerical caveat below.

## What the CI gate can and cannot claim

Dawn on macOS uses **Metal** — the same backend Chrome uses here — so the local
oracle is numerically as close as we can get.

CI has no GPU, so it uses **SwiftShader** (software Vulkan). Tint emits SPIR-V
there and MSL for Metal, on different hardware. Floating-point reassociation and
FMA contraction can differ.

So, precisely:

| The CI gate proves | The CI gate does not prove |
|---|---|
| Indexing, shapes, bounds | Bit-identical floats vs a real GPU driver |
| Control flow and dispatch sizing | Performance of any kind |
| Bind group and buffer layout | Driver-specific behavior |
| Bit-exact dequantization (see below) | |

This must be stated in the packet record rather than discovered later, because
"our kernels are gated in CI" is otherwise a stronger claim than the gate
supports.

## The tolerance, derived

Both the CPU reference and the GPU kernel consume **the same Q4_0 weights**, so
quantization error is common-mode and cancels exactly. The only source of
divergence is f32 accumulation order.

For a dot product of length `K` accumulated in f32, the error bound is
approximately `K · ε · Σ|aᵢbᵢ|` with `ε ≈ 1.19e-7`. At `K = 1024` that is a
relative bound near `1.2e-4`.

The test states the bound as a function of `K` and the operand magnitudes, so
it scales with shape rather than being a constant somebody tuned. **A
divergence above it is a defect, not float noise** — which is the entire reason
the bound is derived rather than chosen.

**The dequantization step is separately gated bit-exact.** Unpacking a Q4_0
block is an integer extract and a single multiply by one fp16 scale — no
accumulation, so exactly one correct IEEE result on any backend, including
SwiftShader. A test-only kernel writes unpacked values and compares byte-for-byte
against BLLM-002's reference. That test-only path must not exist in the
production kernel.

## Acceptance Criteria

- [ ] Dawn builds from a pinned commit via standalone CMake, and the native
      test binary acquires a device on Metal locally.
- [ ] The same binary acquires a device under SwiftShader with no GPU present.
- [ ] The fused dequant-matmul kernel runs natively and produces results within
      the derived tolerance of the CPU reference at every shape the model uses:
      `[1024→2048]`, `[2048→1024]`, `[1024→3072]`, `[3072→1024]`,
      `[1024→151936]`.
- [ ] Dequantized values match BLLM-002's reference **bit-exactly** via the
      test-only unpack path.
- [ ] A deliberately corrupted weight block makes the test fail. A gate nobody
      has seen fail is not evidence.
- [ ] The tolerance is computed from `K` and operand magnitudes in the test
      source, not written as a literal.
- [ ] No `#ifdef` distinguishes native from browser in any WGSL or in
      `src/core/`; `tools/check_boundaries.sh` still passes.
- [ ] The identical kernel runs in the browser and agrees with the native
      result within the same bound, checked manually and recorded.
- [ ] CI runs the GPU tests on every push, with a Dawn build cache, and the
      cold and warm CI build times are recorded.
- [ ] Baseline timings recorded per shape — device, backend, dimensions — as
      the reference BLLM-004 optimizes against.

## Verification Plan

Continuing the ladder from BLLM-002, and note where the oracle changes:

| Rung | Reference | Gate |
|---|---|---|
| Q4_0 unpack | BLLM-002 CPU reference | **byte-exact** |
| matmul at each model shape | our CPU matmul | derived tolerance |
| *(later)* full forward | external implementation | top-1 token match |

The lower rungs prove **our GPU matches our CPU**. Only the top rung, in a later
packet, proves our model matches the world. Conflating the two would let a
consistent misunderstanding of the format pass at every level.

| Test | What it pins |
|---|---|
| `dequant_gpu_test` | Unpacked Q4_0 blocks byte-identical to the CPU reference. |
| `matmul_shape_test` | Each real shape within the derived bound; the bound recomputed per shape. |
| `matmul_corruption_test` | A perturbed weight block fails the gate — proves the gate has teeth. |
| `matmul_lm_head_test` | The 151,936-wide output specifically, where dispatch sizing is most likely to break. |

## C++ Architecture Note

- **Core components touched:** `src/core/gpu/` gains kernel dispatch for the
  fused operation. `src/shaders/` gains the WGSL. `src/core/quant/` is read,
  not modified.

- **Platform wrappers touched:** None. The wrapper stays translation-only; this
  packet adds no browser-facing surface.

- **Public interfaces changed:** Introduced: a dispatch entry taking a weight
  tensor reference (with its quantization parameters, per BLLM-002's
  invariant), an activation buffer, and output dimensions.

- **New files/classes/targets/modules:** `browser_llm_core` gains the kernel
  dispatch in both configurations. A new native-only test target links Dawn.
  `device.cpp` and `self_check.cpp` **stop being wasm-only** — with Dawn
  linked, they compile natively for the first time, closing the source-parity
  gap recorded as residual risk when BLLM-001 was accepted.

- **Dependency edges added or removed:** Dawn, native-only. No new edge in the
  wasm build. Core still depends on `webgpu.h` alone.

- **Ownership/lifetime model:** Unchanged — `UniqueHandle` for every GPU
  resource, ownership flowing through calls as established. Weight buffers are
  owned by the residency layer from BLLM-002 and borrowed by dispatch.

- **Test surface independent of UI wrappers:** This packet's purpose. Kernel
  correctness becomes testable with no browser, which was the standing gap
  since BLLM-001.

- **Why this decomposition is appropriate:** Dawn, the kernel and the gate are
  one intent because none is useful alone. The pieces that *are* separable —
  optimization, further kernels — are excluded.

- **What is intentionally not abstracted:** No matmul abstraction, no kernel
  registry, no backend interface. Shapes are parameters; the kernel is not
  generalized past one model's needs. Prior in-house work found that
  general-purpose ops silently diverge from a specific model's reference, and
  with one model there is nothing to generalize over.

## C++ Performance Note

- **Target hardware/workload:** Chrome and Edge desktop for production; Metal
  locally and SwiftShader in CI for gating. Workload is single matmuls at the
  model's shapes, not a forward pass.

- **Performance budget:** **None set, deliberately.** There is no baseline, and
  a budget invented before one would be a number to rationalize against rather
  than measure against. This packet's output *is* the baseline.

- **Baseline measurement:** Established here. Per shape, recording device,
  backend, dimensions, and the timing boundary used. An unclear boundary
  produces plausible figures that get believed.

- **Hot paths touched:** The fused kernel will become the hottest path in the
  system — every projection and the per-token `lm_head`. It is written naive
  here on purpose; optimizing without the baseline this packet produces would
  violate the performance gate's own sequencing.

- **Allocation/copy/serialization behaviour:** No per-dispatch allocation. The
  weight buffers are already resident from BLLM-002 and are bound, not copied.
  Dequantization is in-register with no intermediate buffer — the invariant
  that keeps quantization worth doing.

- **Concurrency model:** Single-threaded host, unchanged. GPU parallelism is
  the kernel's workgroup structure.

- **Edge AI runtime/backend/configuration:** None adopted. Dawn is a WebGPU
  implementation for testing, not an inference runtime.

- **Before/after measurement plan:** No "after" — nothing is optimized. These
  numbers become BLLM-004's "before".

- **Accuracy/product-contract risk:** The derived tolerance is the contract.
  Its risk is being set too loose — hence deriving it from `K` and operand
  magnitudes rather than choosing it, and hence the corruption test proving the
  gate can fail.

## Records

- Guidelines consulted before design, per `workflow.md`. `cpp-guidelines`:
  SL.con.3 (bounds errors) for buffer indexing in dispatch; I.7 (state
  postconditions) for the kernel's numerical contract; ES.103 (don't overflow)
  for index arithmetic at the 151,936-wide shape.
- `cpp-perf-guidelines`: **the corpus has no GPU material available on this
  server.** Its README advertises a `gpu` category, but the server reports only
  cache-layout, codegen, concurrency, copy-move, embedded, lifetime, memory,
  simd and telemetry — the index is behind the repository. The nearest
  applicable guidance transfers only by analogy: SIMD.5 (gather is a trap;
  restructure to linear loads) and CACHE.3 (contiguity) both map to coalesced
  access in a kernel, but nothing in the corpus speaks to workgroup sizing,
  occupancy or shared memory. **Recorded as a gap:** this packet's performance
  reasoning rests on measurement rather than citation, and the corpus should
  gain GPU material before it can gate GPU work properly.
- Method notes: `research/2026-08-31-model-port-methodology.md` — fixture ladder,
  gate splitting, and the warning against generic ops.
- BLLM-004 follows: optimize the kernel against this packet's baseline, and add
  the remaining kernels on this harness.

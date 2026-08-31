# BLLM-003: GPU compute on quantized weights, gated in CI

**Status:** approved
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

- **Optimization.** No tiling, no occupancy tuning, no shared-memory staging
  (GPU.5 — earn it with measured reuse). The kernel is naive and correct. Not
  deferral-because-hard: the performance gate requires a baseline before
  optimizing, and this packet creates it (GPU.10).

  **Coalesced access is NOT in this exclusion.** GPU.2 is explicit that lane
  mapping is decided *before* tuning, and that the fix is often to transpose at
  load time. Getting it wrong is not a slow kernel to be optimized later — it
  is a weight layout to be re-uploaded, which reaches back into BLLM-002. So
  the lane mapping is a design-time deliverable of this packet.
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
  commit not tag; the numerical caveat below; and **event-loop semantics**.
  Our callbacks currently resolve under the browser's event loop. Native Dawn
  requires the host to drive event processing explicitly, so assumptions that
  compile only under emdawnwebgpu will surface here — an early integration
  task, not a late surprise.

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

The comparison bound must cover **both** computations, because the test
compares two independently rounded results. A one-sided estimate of a single
accumulation is not a valid postcondition (I.7).

So: a γ-style bound for the CPU algorithm covering its products *and* additions,
a second bound for the WGSL algorithm, and the comparison threshold is their
sum — computed from the actual operands, not a constant.

The GPU side needs the wider bound, because **WGSL does not fix an evaluation
order**. The specification permits reassociation and fusion (FMA), does not
specify a rounding direction, and allows flushing eligible subnormal inputs and
results. The kernel's arithmetic is therefore not a single determined sequence,
and the bound must reflect that rather than assume ordered accumulation.

The test must also state its handling of cancellation, overflow, subnormals,
and NaN/infinity, and constrain input ranges so catastrophic cancellation
cannot make a relative bound meaningless.

An earlier draft gave `K · ε · Σ|aᵢbᵢ|` and promoted that approximation into a
hard "anything above is a defect" rule. It bounded one accumulation on one
side, which is the wrong quantity.

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
- [ ] A **negative control** proves the gate detects a real defect. The naive
      version does not work: since both sides consume the same Q4_0 bytes,
      corrupting the shared source is **common-mode and cancels**, and the
      comparison passes. Even GPU-only corruption can stay under tolerance when
      the paired activation is near zero.

      So: compute and retain the CPU oracle from **pristine** bytes, mutate
      **only the uploaded GPU copy**, and choose deterministic scale, nibble and
      activation values whose analytically predicted output delta provably
      exceeds the derived bound. Record the negative-control run as a separate
      artifact, not as a permanently-green test.
- [ ] The **lane→address mapping is written down** for the kernel's hot loads:
      the byte address touched by lanes 0, 1, 2 and 31 when reading Q4_0
      weights, activations, and the block scale. GPU.2's review model. A Q4_0
      block interleaves 32 packed nibbles with one fp16 scale, so the scale
      load is the obvious divergence from a contiguous pattern and needs a
      stated answer, not an accident.
- [ ] The weight layout that mapping implies is **confirmed compatible with
      BLLM-002's upload order**, or BLLM-002 is amended before it is
      implemented. Discovering a transpose is needed after the upload path
      exists means rewriting it (GPU.2: transpose at load time).
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

- **Public interfaces changed:** A **move-only kernel object** owning the
  pipeline, bind group layout and any reusable bindings, constructed once; and
  a dispatch call that *borrows* explicit weight, activation, output and
  workspace buffers and returns a `[[nodiscard]]` typed status.

  An earlier draft named only weights, activation and dimensions — with no
  output destination, no result type, and no stated owner for pipelines, bind
  groups or command objects. That made "no per-dispatch allocation"
  unsupportable: such an API must either allocate internally or rely on ambient
  state (GPU.9 — steady-state allocation stays outside the hot loop).

  Preconditions are part of the interface (I.5): Q4_0 block divisibility,
  buffer byte ranges, binding alignment, dimension compatibility, and device
  limits. Postconditions state what the output buffer contains (I.7). WebGPU
  validation failures surface as observable errors (E.27), and every resource
  releases on **every** failure path (R.1, E.25).

- **New files/classes/targets/modules:** `browser_llm_core` gains the kernel
  dispatch in both configurations. A new native-only test target links Dawn.
  `device.cpp` and `self_check.cpp` **stop being wasm-only** — with Dawn
  linked, they compile natively for the first time, closing the source-parity
  gap recorded as residual risk when BLLM-001 was accepted.

- **Dependency edges added or removed:** Dawn, native-only — declared on
  `browser_llm_core`, **not** on the test executable. Core's public headers
  expose `webgpu.h` types and core compiles natively once Dawn exists, so the
  native header dependency must be **PUBLIC** on the core target and static-link
  requirements must propagate. Tests link the core target rather than
  reconstructing its dependency graph. Attaching Dawn only to the test binary
  would leave core's include and link needs undeclared and satisfied by
  accident. No new edge in the wasm build.

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
  backend, dimensions, build configuration, and the timing boundary — decomposed
  into CPU submission, queue wait, GPU execution and readback rather than one
  wall-clock figure. An unclear boundary produces plausible numbers that get
  believed (GPU.10).

  Kernel timing comes from the **diagnostic** configuration and is labelled as
  such; timestamp queries require the `timestamp-query` feature at device
  acquisition, so an instrumented device is not the device the clean build
  measures. See `research/2026-08-31-measurement-build-configurations.md`.

- **Readback budget (GPU.1): owed by the decode packet, not this one.** GPU.1
  requires a hot-loop scalar readback to carry a written reason and a budget of
  bytes, measured transfer, measured wait, queue or fence waited on, and steps
  delayed. What we have is combined round-trip latency
  (`research/2026-08-31-gpu-readback-round-trip.md`) and a projected
  percentage — not those components, and not measured under real forward-pass
  load. This packet has no decode loop, so claiming the budget here would be
  claiming an artifact we do not have for a loop that does not exist. It is
  recorded as owed by the decode packet.

  What this packet *does* owe: its baseline must separate **CPU submission,
  queue wait, GPU execution, and readback** (GPU.10). A single wall-clock
  number per shape cannot tell BLLM-004 which of those to attack.

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
- `cpp-perf-guidelines`: GPU.1 (keep data on device; every host-device round
  trip needs a budget), GPU.2 (shape data for coalesced lane access **before**
  tuning the kernel), GPU.5 (shared memory only when reuse pays), GPU.6 (batch
  tiny GPU work), GPU.10 (profile with GPU timelines before optimizing).
  **Correction:** an earlier draft of this packet recorded "the corpus has no
  GPU material" as an accepted gap. That was wrong. The corpus has a GPU
  category of 10 guidelines; the MCP server was reading a second checkout at
  `mcp-servers/data/` sitting two commits behind the one where the category
  landed, so `update_guidelines` re-indexed a repo that never moved and
  faithfully reported 77 of 87 guidelines. Diagnosed and fixed by pulling that
  checkout. The lesson is that a tool's answer is not the territory: the
  corpus was two commands away and I did not look.
- Method notes: `research/2026-08-31-model-port-methodology.md` — fixture ladder,
  gate splitting, and the warning against generic ops.
- BLLM-004 follows: optimize the kernel against this packet's baseline, and add
  the remaining kernels on this harness.

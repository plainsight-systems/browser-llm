# BLLM-002: Model selection and weight residency

**Status:** draft
**Change class:** architectural

## Intent

- **What is changing:** The harness gains a target model and gets its weights
  onto the GPU. GGUF container parsing, Q4_0 block layout, the tokenizer, model
  config — and then the part that matters most: **quantized weights streamed
  into GPU buffers, still quantized, and proven byte-identical by readback.**

- **Why the change is necessary:** Weight residency is the highest-risk unknown
  in the project. ~420 MB of weights must land across many buffers, under a
  storage-binding limit that is granted at runtime and varies per device, on a
  platform whose address space is 32-bit. Every kernel decision downstream —
  bind group layout, dequant shader shape, attention tiling — depends on how
  the weights are laid out. Deferring it until the kernels exist would mean
  debugging residency and shaders together.

  An earlier draft of this packet scoped the work CPU-only and justified it as
  "natively testable." That was convenience dressed as principle: it deferred
  the riskiest item and kept the easiest. Sequencing follows risk.

- **Expected behavior changes:** The page gains a load phase — fetch with
  progress, SHA-256 verification, OPFS cache — then reports the parsed model
  summary and a **residency map**: how many buffers, their sizes, which limits
  bound the packing, and confirmation that a sampled block reads back
  byte-identical. Still no inference and no token generated.

- **Guaranteed invariants/contracts:**
  - **Weights reach the GPU quantized.** The CPU never materialises a
    dequantized weight. Dequantization happens in a shader, in BLLM-003. A
    CPU-dequantize-then-upload path would quadruple resident bytes and defeat
    the entire reason for quantization.
  - The GGUF parser treats the file as **untrusted input**. Every offset and
    length is bounds-checked against the actual buffer before use; a malformed
    file yields a named error, never a read outside the mapping (SL.con.3).
  - The whole model is never resident in the WASM heap. Bounded chunks are
    uploaded and released. wasm32 has no large address space to reserve, so
    MEM.7's reserve-and-commit strategy is unavailable to us — streaming is the
    only option, not a preference.
  - Buffer packing is planned from the **granted** device limits, read after
    acquisition, and works at the 128 MiB spec-default storage-binding floor.
  - A SHA-256 mismatch fails the load loudly.

## The decision

**Model: Qwen3-0.6B. Quantization: Q4_0. Container: GGUF. Targets: Chrome and
Edge desktop.**

Read from the model's `config.json`, not assumed:

| | |
|---|---|
| `hidden_size` | 1024 |
| `intermediate_size` | 3072 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` / `num_key_value_heads` | 16 / 8 — GQA, 2:1 |
| `head_dim` | **128** |
| `vocab_size` | 151,936 |
| `rope_theta` | 1,000,000 |
| `rms_norm_eps` | 1e-6 |
| `tie_word_embeddings` | **true** |
| `hidden_act` | silu (SwiGLU) |

**`head_dim` is explicit and is not `hidden_size / num_attention_heads`.**
16 × 128 = 2048 against a hidden size of 1024, so `q_proj` is [1024 → 2048] and
`o_proj` is [2048 → 1024]. Prior in-house work hit this same trap on an
unrelated model. Assert it from the file; never derive it.

**Why this model.** Selected on *kernel surface*, not benchmark quality — the
harness is the demonstration. Qwen3 needs the standard decoder set plus QK-norm.
Rejected `gemma-4-E2B`: a 10.25 GB checkpoint despite the "2B effective" name,
carrying vision and audio towers we would never call. Rejected
`gemma-4-26B-A4B`: MoE optimizes compute, while a browser's binding constraint
is resident bytes — 26B parameters must be resident to route 3.8B of work.
Llama-3.2-1B was runner-up: one fewer kernel, ~280 MB more download. Every
kernel written here transfers unchanged to either.

**Limits floor: none above the WebGPU defaults.** The harness requests the
adapter's advertised maxima, which is always satisfiable and so costs no
portability — but means limits vary per device and are known only after
acquisition. Packing must therefore work at the 128 MiB default and use more
when granted.

## Scope

**In:** GGUF parse, Q4_0 layout, tokenizer, model config, chunked fetch with
SHA-256 and OPFS cache, weight upload to GPU buffers, residency map, readback
verification.

**Explicitly out**, so it cannot drift in:

- Any shader that computes on the weights. Dequantization on the GPU is
  BLLM-003.
- Native Dawn. No kernel yet needs a reference oracle on the GPU.
- Any decode loop, KV cache, or sampling.
- Generic abstractions over quantization formats, tensor types, or backends.
  There is one model and one format; an abstraction over a single
  implementation is ornament.

## The quantized-embedding hazard

Recorded here because it is the defect most likely to reach output silently.

Qwen3-0.6B has `tie_word_embeddings: true`, and the embedding table is
151,936 × 1024 ≈ 155.6M parameters — about **26% of the model**. In GGUF that
table is quantized like any other tensor.

Two failure modes follow, both of which produce **tensors of the correct shape
containing uninterpretable values** — a §3.1 facade that no shape assertion
catches:

1. A token-embedding gather that is not quantization-aware returns packed
   quantized bytes, which the rest of the pipeline treats as floats.
2. The tied output projection, if rebuilt from the embedding tensor without
   carrying its quantization parameters, dispatches an unquantized matmul over
   packed bytes.

Prior in-house work halted a packet on exactly this pair. The mitigation is
structural and starts here: **the tensor index must carry quantization
parameters with every tensor**, so that no consumer can obtain a weight
reference without also obtaining what it takes to interpret it. A design where
quantization is "handled by the loader" and the rest of the pipeline is
quantization-transparent is the design that fails.

## Acceptance Criteria

- [ ] A GGUF file parses to a tensor index and metadata map, verified against
      committed fixtures.
- [ ] Malformed inputs fail with named errors and read nothing out of bounds:
      truncated header, tensor offset past EOF, length overflowing the file,
      bad magic, unknown version, tensor count that cannot fit.
- [ ] Q4_0 dequantization matches independently computed values **bit-exactly**
      on a fixture block. This is the test oracle for BLLM-003's shader.
- [ ] Every tensor in the index carries its quantization parameters; a test
      asserts no accessor returns a weight without them.
- [ ] The tokenizer round-trips a fixture corpus, including multi-byte UTF-8
      and special tokens.
- [ ] Parsed config equals the table above, with `head_dim` asserted as 128
      rather than derived.
- [ ] Weights upload to GPU buffers **while quantized**; a sampled block reads
      back byte-identical to the source file.
- [ ] Packing respects the granted `maxStorageBufferBindingSize`, and a test
      exercises the planner at the 128 MiB spec-default floor, not only at this
      machine's granted value.
- [ ] Peak WASM heap during load stays under a stated bound, asserted by
      instrumentation. The bound must be verified against the **real 420 MB
      file**, not only a fixture — a fixture-only assertion proves nothing
      about the streaming path.
- [ ] SHA-256 mismatch aborts the load with a distinct, user-visible error.
- [ ] The page reports architecture, layer count, tensor count, quantization,
      and the residency map.
- [ ] Everything except upload and readback passes natively in CI.

## Verification Plan

**The fixture ladder.** Built smallest-primitive-first, each rung gated before
the next, so a failure surfaces at the operation that caused it:

```
Q4_0 block dequant  →  tensor index  →  config  →  tokenizer  →  residency
```

Each fixture carries provenance sufficient to re-derive it: model repository,
snapshot hash, the tool and version that produced it, and the source file it
was cut from. A fixture without that is a number nobody can reproduce.

**The gate splits by whether the math is exact.** Our reference will differ
from us in accumulation order and precision, so byte equality is unattainable
in general and pursuing it would burn the packet:

| Byte-exact | Bounded divergence |
|---|---|
| Q4_0 dequantization (integer unpack × scale) | *(nothing in this packet)* |
| Tokenizer round trip | |
| Tensor index, offsets, shapes | |
| Uploaded bytes vs source bytes | |

Everything in this packet is on the exact side — which is a reason to do the
loader before the kernels, where derived tolerances start.

| Test | What it pins |
|---|---|
| `gguf_parse_test` | Header, metadata KV types, tensor index against fixtures. |
| `gguf_malformed_test` | Each malformed case above, named error, no out-of-bounds read. The file arrives from a CDN; this is the test that matters. |
| `quant_q4_0_test` | Block layout and dequantization, bit-exact against independently computed values. |
| `quant_params_test` | No tensor accessor yields a weight without its quantization parameters. |
| `tokenizer_test` | Round trip over a fixture corpus. |
| `model_config_test` | Parsed config equals the known table; `head_dim` asserted explicitly. |
| `residency_plan_test` | Buffer packing at the 128 MiB floor, at a large granted limit, and at a limit that forces a tensor to span buffers. Pure planning arithmetic — no GPU. |

CI must not download 420 MB, so fixtures are small synthetic GGUFs. The real
file is exercised manually and its SHA-256 recorded, including the heap-bound
assertion, which fixtures cannot establish.

## C++ Architecture Note

- **Core components touched:** New under `src/core/`: `gguf/` (reader, types),
  `quant/` (Q4_0 block layout and parameters), `tokenizer/`, `model/` (config),
  `residency/` (buffer planning and upload).

- **Platform wrappers touched:** `src/wasm/bindings.cpp` gains a load entry
  point and a summary reporter. `web/` gains fetch-with-progress, SHA-256 and
  the OPFS cache — in JavaScript, because the browser supplies both primitives.

- **Public interfaces changed:** None existing. Introduced: `gguf::Reader`,
  `gguf::TensorIndex`, `quant::Q4_0Block`, `quant::Params`,
  `tokenizer::Tokenizer`, `model::Config`, `residency::Plan`,
  `residency::upload`.

- **New files/classes/targets/modules:** Parsing, quant, tokenizer, config and
  the residency **planner** compile in both configurations — none need
  `webgpu.h`, so native source parity holds and CI compiles and tests them. The
  upload path itself is wasm-only, like `device.cpp`.

- **Dependency edges added or removed:** None outward. No third-party
  dependency: GGUF is simple enough to parse directly, and vendoring an
  existing loader would defeat the premise.

- **Ownership/lifetime model:** The parser borrows a
  `std::span<const std::byte>` and owns nothing; the caller owns the bytes. The
  tensor index holds offsets and lengths, **not pointers**, so it cannot
  outlive its buffer into undefined behaviour. GPU buffers are `UniqueHandle`
  as everywhere else. Exceptions are off, so errors return as values, following
  `dispatch_count` (E.25).

- **Test surface independent of UI wrappers:** Everything except the upload
  itself. The residency *planner* is deliberately separated from the *upload*
  so the packing arithmetic — where the interesting failures are — is a pure
  function testable at limits this machine does not have.

- **Why this decomposition is appropriate:** Separating plan from upload is the
  load-bearing choice. It lets us test packing at the 128 MiB floor, at a
  spanning boundary, and at limits no available device reports, without a GPU.
  The alternative — packing decided inside the upload call — would make the
  riskiest arithmetic in the packet reachable only through a browser.

- **What is intentionally not abstracted:** No tensor library, no graph
  abstraction, no backend interface, no quantization-format abstraction. One
  model, one format. The model seam is **not** typed on `WGPUBuffer`; a
  backend-typed seam is a known portability trap.

## C++ Performance Note

- **Target hardware/workload:** Chrome and Edge desktop. Workload is model load
  only: fetch, verify, parse, pack, upload. Not inference.

- **Performance budget:** Load is performance-sensitive by this repo's default.
  Header and index parsing is an operation over metadata, not over 420 MB, and
  should complete in **under 100 ms** on the blessed target. Upload throughput
  and SHA-256 over 420 MB are **measured, not budgeted**, in this packet —
  there is no baseline to budget against yet.

- **Baseline measurement:** None exists; this packet establishes it. Recorded
  measurements must name the device, browser build, granted limits, and whether
  the file came from network or OPFS — cold and warm differ by the entire
  download. Timing boundaries must be explicit: a metric that measures the
  wrong interval produces plausible figures that get believed.

- **Hot paths touched:** None. This is the bounded init phase; MEM.9's shape
  applies — allocate during load, run the steady state with a fixed footprint.
  Weight residency established here *is* that fixed footprint.

- **Allocation/copy/serialization behaviour:** The binding constraint. A 420 MB
  model must not be resident in the WASM heap, and wasm32 offers no large
  address space to reserve, so MEM.7's reserve-and-commit approach is
  unavailable. Chunks are uploaded and released; peak heap is asserted by test
  against the real file. Weights are **never** copied into a dequantized form
  on the CPU.

- **Concurrency model:** Single-threaded, unchanged. Fetch and hashing are the
  browser's, in the worker.

- **Edge AI runtime/backend/configuration:** None. No inference runtime exists
  and none will be adopted.

- **Before/after measurement plan:** No "after" — nothing is optimized here.
  These measurements become the baseline BLLM-003 is compared against.

- **Accuracy/product-contract risk from performance optimizations:** None; no
  optimization is performed. The accuracy risk that exists is Q4_0
  dequantization correctness and quantization-parameter propagation, which is
  why the first is pinned bit-exactly and the second has its own test.

## Records

- Guidelines consulted before design, per `workflow.md`. `cpp-guidelines`:
  SL.con.3 (bounds errors), SL.4 (type-safe standard library), P.11
  (encapsulate messy constructs) shaped the span-based, offsets-not-pointers
  parser contract, since the file arrives from a CDN; E.25 governs error
  returns with exceptions off. `cpp-perf-guidelines`: MEM.9 (allocate at init,
  fixed steady state) frames residency as the init-phase footprint; MEM.7
  (reserve address space, commit on demand) was considered and **does not
  apply** — it assumes a 64-bit address space wasm32 does not provide, which is
  itself the reason streaming is mandatory rather than preferred.
- Model sizes measured from the Hugging Face API, not estimated: Qwen3-0.6B
  1.50 GB, Llama-3.2-1B 2.47 GB, Qwen3-1.7B 4.06 GB, gemma-4-E2B 10.25 GB
  (bf16 checkpoints).
- Method notes from prior in-house inference work:
  `research/2026-08-31-model-port-methodology.md`. The readback measurement
  that settles the decode-loop shape: `research/2026-08-31-gpu-readback-round-trip.md`.
- BLLM-003 follows: native Dawn plus the first GPU kernels, using this packet's
  bit-exact dequantization oracle as the correctness reference.

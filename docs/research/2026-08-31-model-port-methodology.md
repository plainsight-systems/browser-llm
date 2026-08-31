# Porting a model to a new backend: method notes

**Date:** 2026-08-31

Notes drawn from reviewing a prior in-house C++ inference harness that ports
transformer models to a GPU backend. That harness targets a different model
family and a different compute stack, so **none of its code applies here** —
what follows is method only, recorded because these are expensive lessons to
re-learn.

Its specifics are internal and deliberately not reproduced.

---

## 1. Build a fixture ladder, smallest primitive first

The most valuable artifact in that codebase was not code — it was a hierarchy
of recorded reference tensors, gated in order:

```
one primitive op        (e.g. positional encoding, in isolation)
  → one attention sub-layer
    → one decoder layer
      → the layer with residuals and norms
        → the full forward
```

Each rung is verified against a reference before the next is attempted. That
ordering is why their failures surfaced *at the operation that caused them*
rather than as "the model emits garbage" at the end.

**For us:** dequantized block → RMSNorm → RoPE → one attention → one layer →
full forward. Build the ladder before the kernels, not after.

## 2. Record enough provenance to re-derive the oracle

Their fixture metadata pins the model snapshot hash, the exact reference
implementation and its commit, the specific source files consulted, and the
versions of every library involved.

A fixture without that is a number nobody can reproduce or re-derive when the
reference moves. Ours should carry the same, for whichever reference we choose.

## 3. "Reusable" ops silently diverge — this is the big one

Two independent cases in that codebase, both caught only by a dedicated
byte-gate on the individual operation:

- A general positional-encoding primitive computed its frequency denominator
  from one dimension where the target model's reference used another. Same
  shape, plausible values, wrong numbers.
- A general masking primitive constructed an explicit mask array, while the
  reference dispatched a *fused* kernel path. Different kernel, no guarantee of
  identical output.

Both were reasonable reuses of an existing, correct-looking primitive. Neither
was findable by inspection — only by gating the op in isolation against the
reference.

**For us:** with exactly one model, there is nothing to generalise over yet.
Write the op the model needs, gate it alone, and resist building a "general"
version until a second model demands one.

## 4. Land sub-layers as gated primitives, not partial forwards

Their sub-layer work lands explicitly *unwired* from the model's forward path,
described as a tested primitive rather than a partial forward — precisely so
that a half-wired path never produces output that looks like progress.

That is the no-facades rule applied to incremental model work, and it is the
right posture for a multi-packet port.

## 5. `head_dim` is not necessarily `hidden_size / num_heads`

Their target model has an explicitly configured head dimension that does not
equal hidden size divided by head count. So does ours: Qwen3-0.6B configures
`head_dim: 128` against `hidden_size: 1024` and `num_attention_heads: 16`,
where the naive derivation gives 64.

Two unrelated models, the same trap. **Read it from the file; never derive it.**

## 6. Claim performance narrowly, and name the residual

Their sub-layer headers claim a specific allocation property, then state
plainly what is *not* claimed and which transient allocations remain and why —
recorded as a residual rather than rounded up into a stronger claim.

That is the standard to match: claim exactly what was verified, name what was
not.

## 7. Structural patterns worth copying

- Cohesive weights and dimensions travel as **one struct**, not a long
  parameter list.
- Each sub-layer exposes a **closed error enum**, with errors from composed
  primitives translated into it at the boundary, so callers see one vocabulary.
- Guidelines are cited **in the header**, next to the decision they justify,
  rather than in a separate document nobody reads alongside the code.

---

## What explicitly does not carry over

- **The model architecture.** Theirs is a different family with structural
  features ours does not have. Copying its layer shape would add machinery
  Qwen3-0.6B does not contain.
- **Every backend primitive.** That harness composes vendor-provided fused
  operations — fused attention in particular. **WebGPU provides none of these.**
  Every operation is a WGSL kernel we write, test and optimise ourselves.

  This inverts the effort profile. Their hard problem was *matching* a
  reference operation for operation; ours is *implementing* the operations at
  all. Any conclusion they reached about where time goes is therefore theirs,
  not automatically ours.
- **The quantization layout.** Their stack keeps quantization scales in
  tensors separate from the packed weights. GGUF Q4_0 interleaves one scale
  with each block of 32 weights inside a single tensor. Different layout,
  different dequantization kernel, different loader.
- **Host/device memory assumptions.** That harness runs where host and device
  share memory, so it has no transfer layer at all. We have an explicit
  boundary and must design for it from the start.

---

## The adaptation that matters most: our gate cannot be byte-equal

That harness gates on **byte equality** against its reference. This is
achievable only because the port and the reference use *the same underlying op
library* — matching call for call yields bit-identical results.

We have no such luxury. Any reference we pick will differ from our WGSL in
accumulation order, precision, and kernel structure. **Byte equality is not
attainable**, and pursuing it would burn the packet.

So the gate splits by whether the math is exact:

| Exact — gate byte-equal | Floating-point — gate by bounded divergence |
|---|---|
| Q4_0 block dequantization (integer unpack × scale) | matmul / attention output |
| Tokenizer encode/decode round trip | RMSNorm, RoPE, softmax |
| `argmax` token selection | accumulated layer output |
| Tensor index and shape derivation | end-to-end logits |

For the right-hand column the tolerance must be **derived** — from accumulation
width and element count — not chosen because it made a test pass. And the
**top-1 token must match exactly** even where the logits do not; that is the
gate that actually determines output quality, and it is a weaker but far more
meaningful contract than a float comparison.

---

## Implications for BLLM-002 / BLLM-003

1. Build the fixture ladder first, with full provenance metadata.
2. Byte-gate the dequantizer; tolerance-gate everything downstream, tolerance
   derived, top-1 match asserted separately.
3. Write no generic ops — there is one model to satisfy.
4. Land each sub-layer as a gated primitive, unwired, until the forward packet.
5. Assert `head_dim == 128` from the file; never derive it.
6. Consider C++23 for `std::expected` — verified available in both our host
   clang and Emscripten toolchains (`__cpp_lib_expected == 202211`). It gives
   the closed-error-at-the-boundary pattern directly. Note the same bump does
   *not* provide `std::move_only_function`, which libc++ still lacks.
7. Do not type the model seam on `WGPUBuffer`. A backend-typed seam is a known
   portability trap.

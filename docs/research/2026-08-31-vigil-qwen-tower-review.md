# Reviewing vigil's Qwen tower: what transfers to browser-llm

**Date:** 2026-08-31
**Reviewed:** `vigil/backends/mlx/qwen35_*.{h,cpp}`, `qwen3_5_moe_*.{h,cpp}`,
`model_variant.h` (~15,100 lines), plus `tests/fixtures/qwen35_text_forward/`.

## First, the thing that must not be assumed

**Vigil's Qwen tower is a different model architecture from ours.** It targets
Qwen3.5 / Qwen3-Next (`mlx-community/Qwen3.5-4B-4bit`, `config_model_type:
qwen3_5`). We selected Qwen3-0.6B. These share a vendor and almost nothing else:

| | vigil's Qwen3.5 / Qwen3-Next | our Qwen3-0.6B |
|---|---|---|
| Attention | **Hybrid** — 8 of 32 layers full-attention, the rest Gated-DeltaNet linear attention | All layers full attention |
| Extra machinery | `conv_ring`, `gated_delta`, `hybrid_cache`, MoE router, vision module | none |
| Q projection | **Doubled** — `n_heads*head_dim*2`, split per-head into queries **and a gate** | plain |
| Output | `o_proj(out * sigmoid(gate))` — sigmoid output gate | plain `o_proj` |
| RoPE | **Partial** — `rotary_dim = head_dim * 0.25` (64 of 256 dims rotate) | full |
| SDPA scale | `head_dim**-0.5` | same |
| Sparsity | MoE | dense |

So `qwen35_full_attn.cpp`, `qwen35_linear_attn.cpp`, `qwen35_gated_delta.cpp`,
`qwen35_conv_ring.cpp`, `qwen35_hybrid_cache.cpp` and both MoE modules are
**not a template for our decoder layer**. Copying their attention shape would
give us a gate and a partial RoPE the model does not have.

What is worth taking is almost entirely **method**, not code.

---

## What transfers

### 1. The fixture ladder — the single most valuable artifact

`tests/fixtures/qwen35_text_forward/` is a hierarchy, smallest primitive first:

```
rope64.safetensors        → one op, in isolation
full_attn.safetensors     → the self_attn sub-layer
decoder_layer.safetensors → one layer
full_layer.safetensors    → the layer with residuals and norms
full_model.safetensors    → end to end
sanitize_probe.json / metadata.json
```

Each rung is gated independently against a pinned reference before the next is
attempted. That is why their bugs were caught at the op that caused them rather
than as "the model outputs garbage."

**We should build the same ladder** for Qwen3-0.6B: dequant block → RMSNorm →
RoPE → one attention → one layer → full forward.

### 2. Provenance recorded well enough to re-derive the oracle

`metadata.json` pins the reference exactly:

```json
"model_repo": "mlx-community/Qwen3.5-4B-4bit",
"snapshot_sha": "0e7ffd5c...",
"mlx_lm_commit": "ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd",
"reference_files": ["mlx_lm/models/qwen3_5.py", ...],
"versions": { "mlx": "0.31.2", ... }
```

Model snapshot, reference implementation commit, the specific source files, and
library versions. A fixture without this is a number nobody can reproduce.

### 3. "Reusable" ops silently diverge — two documented cases

This is the highest-value warning in the whole tower, and both cases are
recorded in `qwen35_full_attn.h` as **evidence-based corrections**:

- **RoPE.** Vigil had a general `partial_rope_op` (ProportionalRoPE, frequency
  denominator = `head_dim`). Qwen needs `nn.RoPE(dims=64)` — denominator =
  `rotary_dim`. Not byte-equal. Caught by a dedicated `rope64` byte-gate.
- **Causal mask.** The reference dispatched the *fused* causal-mask SDPA kernel
  (`mask="causal"`, no array). Vigil's general `causal_mask_op` builds an array
  and hits a different kernel — "not guaranteed byte-identical."

Both were plausible reuses of an existing, correct-looking primitive that were
wrong for this model. Neither would have been caught by inspection.

### 4. Land sub-layers as tested primitives, not partial forwards

`qwen35_full_attn.h` is explicit: the packet lands the sub-layer as "a TESTED
PRIMITIVE (same posture as 256-a/b/c/d), not a partial model forward (no §3.1
facade)," and states plainly that it is **not** wired into
`forward_with_cache`/`decode_step` yet. A half-wired forward that produces
output would be the facade; an unwired, gated primitive is not.

### 5. `head_dim` is decoupled from `hidden/heads` — independently confirmed

Their header notes "head_dim is config-driven and decoupled from hidden/heads
(256 != 2560/16)." We found the same for Qwen3-0.6B (128 ≠ 1024/16) reading
`config.json`. Two models, same trap, arrived at independently.

### 6. Narrow, honest performance claims

`qwen35_full_attn.h` claims MEM.9 **narrowly** and then states what is *not*
claimed: the KV arena does not grow on decode append, but this is "NOT a full
'no C++ heap allocation on the decode step'," and the per-call transients "ARE
bounded... and are NOT pre-sizable without diverging — that is the RESIDUAL,
recorded honestly."

That is the standard to match: claim exactly what was verified, name the
residual.

### 7. Structural patterns worth copying

- Cohesive weights and dims travel as **one struct** (`QwenAttnLoadParams`),
  not a long argument list — C.1 / I.23.
- A **closed error enum** per sub-layer, with the composed primitives' errors
  translated into it at the boundary "so callers see ONE closed vocabulary."
- Guidelines cited **in the header**, next to the decision they justify.

---

## What does not transfer

### The architecture

Covered above. Their attention is a different operator.

### Every MLX primitive

Vigil composes `mx::fast::rope`, `mx::fast::scaled_dot_product_attention`
(fused, with an internal GQA `repeat_kv`), `mx::quantized_matmul`,
`mx::fast::rms_norm`. **We have none of these.** There is no fused SDPA in
WebGPU. Every one of those is a WGSL kernel we write, test, and optimise
ourselves.

This inverts the effort profile. Vigil's hard problem was *matching* a
reference op-for-op; ours is *implementing* the ops at all. Their first harness
audit concluded "the kernels extract the silicon; the loop around them does
not" — that conclusion is theirs, not automatically ours, because their kernels
came from MLX and ours will not.

### The quantization format

MLX block-quant stores `weight` (packed `uint32`), `scales` and `biases` as
**separate tensors**, with `bits` and `group_size` from config. GGUF Q4_0
interleaves a scale with its 32 weights inside **one** tensor. Different layout,
different dequant kernel, different loader. `QuantParams{bits, group_size}`
does not describe Q4_0.

### safetensors, and `mx::` types in the seam

Their model seam is typed on `mlx::core::array` — the second harness audit flags
this as a cross-silicon readiness item. We are pre-committed to `webgpu.h`
buffers; the equivalent mistake for us would be typing our seam on `WGPUBuffer`.

### Apple unified memory

Their audit's biggest hidden finding: "no host↔device transfer code anywhere
because Apple Silicon has no VRAM boundary... an assumption baked into the
*absence* of an abstraction." We have that boundary explicitly, so this is a
warning about a gap we cannot accidentally inherit — but must design for.

---

## The adaptation that matters most: our gate cannot be byte-equal

Vigil gates on **byte equality** against mlx-lm. That is achievable because the
port and the reference use *the same op library* — matching `mx::` call for
`mx::` call gives bit-identical results.

We have no such luxury. Our reference would be llama.cpp or transformers, whose
matmul accumulation order, precision, and kernels differ from whatever we write
in WGSL. **Byte equality against them is not attainable**, and pursuing it would
burn the packet.

So the gate has to be split by whether the math is exact:

| Exact — gate byte-equal | Floating-point — gate by bounded divergence |
|---|---|
| Q4_0 block dequantization (integer unpack × fp16 scale) | matmul / attention output |
| Tokenizer encode/decode | RMSNorm, RoPE, softmax |
| `argmax` token selection | accumulated layer output |
| Tensor index / shape derivation | end-to-end logits |

For the right-hand column the tolerance must be **derived and justified** — from
the accumulation width and element count — not picked because it made the test
pass. And the top-1 token should match exactly even where the logits do not,
which is the gate that actually matters for output quality. Vigil uses exactly
this idea in its "R8 Tier-3 top-1 match vs Python" contract.

---

## Concrete implications for BLLM-002 / BLLM-003

1. **Build the fixture ladder first**, with full provenance metadata. It is the
   deliverable that makes every later kernel debuggable.
2. **Byte-gate the dequantizer**; tolerance-gate everything downstream, with the
   tolerance derived and the top-1 match asserted separately.
3. **Write no generic ops.** Vigil's two documented failures were both reuses of
   a correct-looking general primitive. With one model there is nothing to
   generalise over yet.
4. **Land each sub-layer as a gated primitive**, unwired, until the full forward
   packet — no partially-wired forward that produces output.
5. **Assert `head_dim == 128` from the file**, never derive it.
6. **Consider C++23 for `std::expected`** — verified available in both our host
   clang and Emscripten toolchains (`__cpp_lib_expected == 202211`). It would
   give us vigil's closed-error-at-the-boundary pattern directly. Note this is
   the same standard bump that does *not* buy `std::move_only_function`, which
   libc++ still lacks.
7. **Do not type the model seam on `WGPUBuffer`** — the mistake their audit
   found in their own `mlx::core::array`-typed seam.

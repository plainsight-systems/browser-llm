# GPU readback round trip: what a per-token read actually costs

**Date:** 2026-08-31
**Device:** Chrome on Apple silicon (`apple` / `metal-3`), WebGPU via emdawnwebgpu.
**Code:** `src/core/gpu/readback_bench.{h,cpp}`, run from the page with `?bench`.

## Why this was measured before designing the decode loop

Vigil's first harness audit concluded that its ~1.6–2× shortfall was
**orchestration, not kernels** — "the kernels extract the silicon; the loop
around them does not." Its single v1.0-blocking finding was a per-token
`.item<>()` forcing a device→host sync every token, *on unified memory where
there is no bus at all*.

WebGPU's equivalent is worse on paper: reading a sampled token back means
`wgpuBufferMapAsync`, an async round trip through the JavaScript event loop.
If that cost is large, a decode loop that reads its token back each step is
capped by it regardless of kernel quality — and the whole architecture has to
change to hide it.

So the number was measured rather than assumed.

## Method

200 iterations of each phase, against a deliberately trivial dispatch (64
elements, one workgroup) so the measurement isolates round-trip overhead
rather than compute:

- **sequential** — submit, map, read, unmap; each iteration waits for the last.
  This is exactly the per-token decode shape.
- **batched** — 200 dispatches submitted, then a single readback.

Timing comes from an injected clock (`emscripten_get_now` supplied by the
wrapper), so `core/` keeps no platform dependency. The mapped value is read
rather than discarded.

## Result

Two consecutive runs, same machine:

| | run 1 | run 2 |
|---|---|---|
| sequential min | 0.300 ms | 0.300 ms |
| **sequential median** | **0.500 ms** | **0.500 ms** |
| sequential p95 | 0.800 ms | 0.800 ms |
| sequential max | 6.500 ms | 6.300 ms |
| sequential mean | 0.594 ms | 0.583 ms |
| batched, 200 dispatches + 1 readback | 8.70 ms | 9.90 ms |
| batched, per dispatch | 0.0435 ms | 0.0495 ms |

Stable to the reported precision across runs.

## What it means

**A serialized GPU round trip costs ~0.5 ms median, ~0.8 ms at p95.** Submit
cost alone is ~0.045 ms, so nearly all of it is the map and the event-loop hop.

That caps a read-back-per-token design at roughly **2000 tokens/s** — an order
of magnitude above any rate a 0.6B model will reach in a browser. At a
plausible 20–50 tok/s the per-token round trip is **1–3% of the token budget**,
not a ceiling.

**So vigil's finding does not transfer as a blocker.** It was v1.0-blocking
there because their token time was ~12 ms (79 t/s) and the sync both cost a
meaningful fraction *and* prevented compute/host overlap. Here the absolute
cost is 0.5 ms against a token that will take far longer.

The 10–11× "cost of serializing" in the page output is measured against a
trivial dispatch and **overstates the case for real work**. The transferable
figure is the absolute 0.5 ms, not the ratio: once a token costs 20 ms of real
compute, the round trip is additive overhead, not a multiplier.

## The decision this settles

**Read the sampled token back per step. Do not build a speculative pipeline.**

This is the inversion worth recording. Vigil's most serious correctness defect
— a P1 that corrupted turn 2 onward of every multi-turn conversation — was
introduced *by* the async pipeline it added to recover throughput: the
speculative `decode_step` wrote the stop token's KV entry into a persistent
cache. Three review rounds and an acceptance missed it, and the regression
suite was structurally blind because `max_new_tokens = 8` meant turns never
terminated naturally.

We have just measured that we do not need that complexity. Taking it anyway
would import the risk without the reason.

If a future model or a faster device changes the arithmetic, this measurement
is cheap to re-run and the decision can be revisited on evidence.

## Caveats

- One device, one browser. The blessed targets include Edge and other GPUs;
  the absolute number will differ, though the order of magnitude is unlikely
  to change enough to matter.
- The `max` of ~6.5 ms shows a real tail — scheduling or GC. Irrelevant at 1–3%
  of a token budget, but it is not a flat 0.5 ms.
- Measured with a trivial dispatch. What this establishes is the *overhead*
  floor, not the behavior when a real forward pass is in flight; overlap
  behavior under load is still unmeasured.

## Lessons taken from vigil that this does **not** settle

Recorded so they are not lost, and are owed by the packets that touch them:

- Greedy sampling must use `argmax` directly. Vigil paid a full-vocab
  `logsumexp` every token over a 262K vocabulary when subtracting a per-row
  constant cannot change an argmax. Qwen3-0.6B's vocab is 151,936.
- Detokenization must be incremental. Vigil's `decode_delta` re-decoded the
  entire sequence per token — O(N²).
- There must be exactly one decode loop. Vigil had four, so every fix landed
  four times or drifted.
- **Quantized embedding with a tied lm_head breaks any "loader-only" quant
  assumption.** Vigil halted a packet on this: a gather over packed `uint32`
  weights is not quantization-aware, and the tied head rebuilt a projection
  that discarded quant state — producing tensors of the right shape with
  uninterpretable content. Qwen3-0.6B has `tie_word_embeddings: true` and its
  embedding is ~26% of parameters, so this applies directly.
- Timing boundaries must be explicit. Vigil's `prefill_tokens_per_s` measured
  the wrong interval and produced implausible figures that were believed for a
  full packet cycle.

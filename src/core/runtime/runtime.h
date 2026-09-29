#pragma once

namespace bllm::runtime {

// Axis M: changes with a new stage in the generation loop.
//
// The turn loop. For each turn: diff the incoming tokens against the cache,
// truncate, prefill the remainder, then decode and sample one token per step
// until a stop token, the context offered, or cancel.
//
//   - Driven by callbacks. Each step submits its GPU work and resumes when the
//     readback completes; nothing blocks.
//   - Each token's text is emitted as the stream decoder completes it: one
//     crossing of the boundary per token.
//   - Cancel takes effect at the next step boundary. The cache keeps every
//     token already computed, so the next turn's diff reuses them.
//   - The per-token path allocates nothing; every buffer it touches was sized
//     by the residency plan at load.

}  // namespace bllm::runtime

#pragma once

#include <cstdint>
#include <span>

#include "core/policy/policy.h"
#include "core/tokenizer/tokenizer.h"

namespace bllm::sampler {

// Axis F: changes with a new sampling method. The settings a model samples
// with are policy, and are passed in.
//
// Contract 10: the sampler.
//
//   - The GPU reduces the logits to the top-k candidates, and only those are
//     read back. A full vocabulary of f32 logits is 0.5 to 1 MiB per token for
//     the models this harness targets.
//   - Randomness is a pure function of the seed and the token's position, with
//     no generator state, so a seed reproduces a run and any one step can be
//     replayed alone.
//   - Temperature zero chooses the first candidate. That is for comparing
//     against a reference in tests; model cards warn against greedy decoding,
//     and the product sampler is stochastic.

struct Candidate {
    tokenizer::TokenId token;
    float logit;
};

// Preconditions: `candidates` is non-empty and sorted by descending logit.
// Applying settings.top_k and the other cut-offs is the sampler's job.
[[nodiscard]] tokenizer::TokenId sample(std::span<const Candidate> candidates,
                                        const policy::SamplingSettings& settings,
                                        policy::Seed seed, std::uint32_t position) noexcept;

}  // namespace bllm::sampler

#pragma once

#include <cstdint>

namespace bllm::policy {

// Contract 12: policy.
//
// A model's measured configuration, from its entry in web/models.json. The
// values change when a model is measured (axis J); this header changes only
// when what can be configured changes.
//
// Two parts cross the boundary at different times. Cache precision is fixed at
// load. Sampling settings depend on the mode a turn runs in, such as thinking
// or not, so the JavaScript side resolves them for the turn and passes them
// with each generate, together with the seed.
//
// An unmeasured model runs on the defaults below and is shown as unmeasured.
// The context offered is not policy: it is derived from the file, the device
// budget and the cache precision.

enum class CachePrecision {
    F16,
    BF16,
    Q8_0,
};

struct LoadPolicy {
    CachePrecision cache_precision = CachePrecision::F16;
};

// Defaults are llama.cpp's, the most widely exercised settings for models
// without a card of their own.
struct SamplingSettings {
    float temperature = 0.8f;
    std::uint32_t top_k = 40;
    float top_p = 0.95f;
    float min_p = 0.05f;
};

// The randomness of a run. Its own type, so a seed cannot be passed where a
// count or a position is meant.
enum class Seed : std::uint64_t {};

struct TurnPolicy {
    SamplingSettings sampling;
    Seed seed;
};

}  // namespace bllm::policy

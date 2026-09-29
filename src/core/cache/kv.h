#pragma once

#include <cstdint>
#include <vector>

#include "core/model/model_description.h"
#include "core/policy/policy.h"
#include "core/residency/plan.h"

namespace bllm::cache {

// Axis G: changes with a new cache or context mechanism.
//
// Contract 8: the KV cache.
//
//   - Full context length for every layer, including sliding-window layers.
//     The window is applied by attention, not by storage, so truncation
//     resets a counter and never requires recomputing evicted entries.
//   - Capacity is the context offered. Each layer's window comes from the
//     model description, storage precision from load policy, and packing from
//     the format for that precision.
//   - Storage is planned by the residency plan and created by upload. The
//     cache names its buffers by plan index and holds no GPU object, so its
//     bookkeeping is tested without a device.
//   - Attention reads a layer's window and writes the step's new entries; the
//     runtime then advances the cache by the step's token count.

class KvCache {
public:
    KvCache(const model::ModelDescription& model, const residency::ResidencyPlan& plan,
            policy::CachePrecision precision, std::uint32_t capacity);

    // Tokens whose keys and values are held.
    [[nodiscard]] std::uint32_t length() const noexcept;
    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] policy::CachePrecision precision() const noexcept;

    // Precondition: tokens <= length().
    void truncate(std::uint32_t tokens) noexcept;

    // Precondition: length() + tokens <= capacity().
    void advance(std::uint32_t tokens) noexcept;

    // Precondition: `layer` is a layer of the model.
    [[nodiscard]] const residency::PlannedCacheLayer& layer(model::LayerIndex layer) const noexcept;

private:
    std::vector<residency::PlannedCacheLayer> layers_;
    std::uint32_t length_ = 0;
    std::uint32_t capacity_;
    policy::CachePrecision precision_;
};

}  // namespace bllm::cache

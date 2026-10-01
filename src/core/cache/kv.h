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
//   - A full-attention layer holds every token of the context offered. A
//     sliding-window layer holds a ring of the slots the residency plan gives
//     it: its window, a prefill block and the policy's rollback reserve.
//     Position p lives in slot p mod slots, in every layer.
//   - Truncation resets a counter and moves no data. A ring has overwritten
//     what lies further back than its slots, and recomputing those entries
//     needs the ones before them, back to the first token. So a rollback
//     within the reserve keeps the cache, and a deeper one empties it.
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

    // Keeps the first `tokens` tokens and returns how many were kept: `tokens`
    // when every sliding-window layer still holds the window before that
    // position, otherwise 0, and the caller prefills from the first token.
    // Precondition: tokens <= length().
    [[nodiscard]] std::uint32_t truncate(std::uint32_t tokens) noexcept;

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

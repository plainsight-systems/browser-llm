#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/index.h"
#include "core/model/model_description.h"
#include "core/policy/policy.h"
#include "core/residency/weight_view.h"

namespace bllm::residency {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// Contract 5: the residency plan. A pure function. Tensor index, model
// description, device limits and load policy in; buffers, each tensor's place
// in them, the cache, the working buffers, and the context offered, out. No
// GPU, no fetch, no browser. Preflight's Fit stage runs it before any weight
// byte is downloaded, and upload carries it out afterwards.
//
//   - Three pools, never sharing a buffer, because their lifetimes differ
//     (GPU.9): the weights, packed in file order into as few buffers as the
//     limits allow; the KV cache, keys and values per layer at the context
//     offered; and one set of working buffers that every layer reuses.
//   - Limits are the ones the device granted, never the adapter's advertised
//     maxima (WASM.10). Packing works at WebGPU's default limits.
//   - A weight larger than one storage binding is split by rows. Every offset
//     is aligned to the device's storage-offset alignment, and every range is
//     padded to a multiple of 4 bytes, as a storage binding requires.
//   - The working buffers hold a prefill block of kPrefillBlock tokens at f32.
//     Attention never stores a block-by-context matrix of scores: at 512
//     tokens, 32 heads and a 40,000-token context that is 2.6 GB. Kernels work
//     within these buffers.
//   - The context offered is the largest that fits the memory budget and the
//     binding limit, capped at the context the model was trained for. Every
//     layer's cache is full length, sliding-window layers included.
//   - The fit counts every tensor. An output head stored as a byte-for-byte
//     copy of the token embedding — tied weights written twice — is marked a
//     candidate duplicate and given buffers of its own, which upload does not
//     create once it confirms the bytes match; a fit never depends on sharing
//     that has not been confirmed.

// The block of tokens one prefill step processes, and the shortest context
// worth offering.
inline constexpr std::uint32_t kPrefillBlock = 512;

// The limits the device was granted. How many bytes the plan may place on the
// device is not among them — WebGPU does not report device memory — and comes
// from load policy instead.
struct DeviceLimits {
    std::uint64_t max_buffer_size;
    std::uint64_t max_storage_binding_size;
    std::uint32_t storage_offset_alignment;
};

enum class Pool {
    Weights,
    Cache,
    Scratch,
};

struct PlannedBuffer {
    Pool pool;
    std::uint64_t size;
};

// A bound range of one planned buffer.
struct BufferRange {
    BufferIndex buffer;
    std::uint64_t offset;
    std::uint64_t length;
};

struct PlannedTensor {
    gguf::TensorId tensor;
    WeightView view;
    // Set when this tensor may be a byte-for-byte copy of an earlier one.
    std::optional<gguf::TensorId> candidate_duplicate_of;
};

// One layer's cache storage, at the context offered.
struct PlannedCacheLayer {
    BufferRange keys;
    BufferRange values;
};

// A working buffer, named for what it holds.
struct PlannedScratch {
    std::string_view purpose;
    BufferRange range;
};

struct ResidencyPlan {
    std::vector<PlannedBuffer> buffers;
    std::vector<PlannedTensor> tensors;
    std::vector<PlannedCacheLayer> cache;
    std::vector<PlannedScratch> scratch;
    std::uint32_t context_offered = 0;
    // Bytes in each pool, alignment padding included, and their sum. On
    // ExceedsBudget, total_bytes is what the shortest context worth offering
    // would need, so a rejection can state needed against available.
    std::uint64_t weight_bytes = 0;
    std::uint64_t cache_bytes = 0;
    std::uint64_t scratch_bytes = 0;
    std::uint64_t total_bytes = 0;
};

enum class PlanError {
    Ok,
    // Weights and working buffers leave too little of the budget for a
    // kPrefillBlock-token cache.
    ExceedsBudget,
    // One row of a weight is wider than a storage binding, so no split by
    // rows can place it. The subject names the tensor.
    RowExceedsBinding,
    // A working buffer is wider than a binding. The subject names it.
    ScratchExceedsBinding,
    // The cache precision stores whole blocks, and a layer's head dimension
    // is not a whole number of them. The subject names the layer.
    UnsupportedCachePrecision,
    // A size that overflows 64 bits: a file no device could hold.
    Overflow,
};

struct PlanResult {
    PlanError error = PlanError::Ok;
    std::string subject;

    [[nodiscard]] bool ok() const noexcept { return error == PlanError::Ok; }
};

[[nodiscard]] PlanResult plan_residency(const gguf::TensorIndex& index,
                                        const model::ModelDescription& model,
                                        const DeviceLimits& limits,
                                        const policy::LoadPolicy& policy,
                                        ResidencyPlan& out);

}  // namespace bllm::residency

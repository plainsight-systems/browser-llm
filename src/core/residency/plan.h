#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "core/gguf/index.h"
#include "core/model/model_description.h"
#include "core/policy/policy.h"
#include "core/residency/weight_view.h"

namespace bllm::residency {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// Contract 5: the residency plan. A pure function. Tensor index, model
// description, device limits and load policy in; buffers, and each tensor's
// place in them, out. No GPU, no fetch, no browser. Gate 4 runs it before any
// weight byte is downloaded, and upload carries it out afterwards.
//
//   - Every tensor and the KV cache are placed, within the memory budget in
//     policy. The cache is sized from the model description and the cache
//     precision in policy.
//   - Limits are the ones the device granted, never the adapter's advertised
//     maxima (WASM.10). Packing works at WebGPU's default limits.
//   - A tensor larger than one storage binding is split by rows.
//   - Every offset is aligned to the device's storage-offset alignment. GGUF
//     aligns to 32 bytes; WebGPU requires 256 by default.
//   - The fit check counts every tensor. Tensors with the same format, shape
//     and length are marked as candidate duplicates; upload shares their
//     storage only after confirming the bytes match, so a fit never depends on
//     sharing that has not been confirmed. A candidate is placed in a buffer
//     of its own, which upload does not create when the match is confirmed.

// The limits the device was granted. How many bytes the plan may place on the
// device is not among them — WebGPU does not report device memory — and comes
// from load policy instead.
struct DeviceLimits {
    std::uint64_t max_buffer_size;
    std::uint64_t max_storage_binding_size;
    std::uint32_t storage_offset_alignment;
};

struct PlannedBuffer {
    std::uint64_t size;
};

struct PlannedTensor {
    gguf::TensorId tensor;
    WeightView view;
    // Set when this tensor may be a byte-for-byte copy of an earlier one.
    std::optional<gguf::TensorId> candidate_duplicate_of;
};

// One layer's cache storage, at full context length.
struct PlannedCacheLayer {
    BufferIndex keys;
    BufferIndex values;
};

struct ResidencyPlan {
    std::vector<PlannedBuffer> buffers;
    std::vector<PlannedTensor> tensors;
    std::vector<PlannedCacheLayer> cache;
    // Bytes the plan needs. Filled on success and on ExceedsBudget, so a
    // rejection can state needed against available.
    std::uint64_t total_bytes = 0;
};

enum class PlanError {
    Ok,
    ExceedsBudget,
    // A single row is wider than one storage binding, so no split can place it.
    RowExceedsBinding,
};

[[nodiscard]] PlanError plan_residency(const gguf::TensorIndex& index,
                                       const model::ModelDescription& model,
                                       const DeviceLimits& limits,
                                       const policy::LoadPolicy& policy,
                                       ResidencyPlan& out);

}  // namespace bllm::residency

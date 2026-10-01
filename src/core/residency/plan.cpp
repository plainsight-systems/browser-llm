#include "core/residency/plan.h"

#include <algorithm>
#include <array>

#include "core/gguf/checked.h"

namespace bllm::residency {
namespace {

using gguf::checked_add;
using gguf::checked_mul;

constexpr std::uint64_t kBindingGranule = 4;   // a storage binding's size is a multiple of 4
constexpr std::uint64_t kF32Bytes = 4;

// v rounded up to a multiple of `to`, a power of two. False on overflow.
bool round_up(std::uint64_t v, std::uint64_t to, std::uint64_t& out) {
    std::uint64_t sum = 0;
    if (!checked_add(v, to - 1, sum)) return false;
    out = sum & ~(to - 1);
    return true;
}

PlanResult failure(PlanError error, std::string subject = {}) {
    return PlanResult{error, std::move(subject)};
}

// Places ranges into the buffers of one pool, filling a buffer before opening
// the next. Every range starts at an aligned offset and is padded to the
// binding granule.
class Packer {
public:
    Packer(std::vector<PlannedBuffer>& buffers, Pool pool, const DeviceLimits& limits)
        : buffers_(buffers), pool_(pool), limits_(limits) {}

    // Places `length` bytes. `alone` gives the range a buffer nobody else
    // shares. Precondition: the padded length fits in one buffer.
    bool place(std::uint64_t length, bool alone, BufferRange& out) {
        std::uint64_t padded = 0;
        if (!round_up(length, kBindingGranule, padded)) return false;
        std::uint64_t offset = 0;
        if (open_ && !alone) {
            if (!round_up(buffers_.back().size, limits_.storage_offset_alignment, offset)) return false;
            if (offset > limits_.max_buffer_size || padded > limits_.max_buffer_size - offset) open_ = false;
        }
        if (!open_ || alone) {
            buffers_.push_back({pool_, 0});
            offset = 0;
        }
        buffers_.back().size = offset + padded;
        out = {static_cast<BufferIndex>(buffers_.size() - 1), offset, length};
        open_ = !alone;
        return true;
    }

private:
    std::vector<PlannedBuffer>& buffers_;
    Pool pool_;
    DeviceLimits limits_;
    bool open_ = false;
};

// The most one range may hold: it must fit both a binding and a buffer.
std::uint64_t range_limit(const DeviceLimits& limits) {
    return std::min(limits.max_storage_binding_size, limits.max_buffer_size) & ~(kBindingGranule - 1);
}

bool is_tied_copy(const gguf::TensorIndex& index, const model::ModelDescription& model,
                  std::size_t tensor) {
    if (!model.output_head || static_cast<std::size_t>(*model.output_head) != tensor) return false;
    const gguf::TensorEntry& head = index.tensor(*model.output_head);
    const gguf::TensorEntry& embedding = index.tensor(model.token_embedding);
    return head.type == embedding.type && head.data_length == embedding.data_length &&
           std::equal(std::begin(head.dimensions), std::end(head.dimensions),
                      std::begin(embedding.dimensions));
}

PlanResult place_weights(const gguf::TensorIndex& index, const model::ModelDescription& model,
                         const DeviceLimits& limits, ResidencyPlan& out) {
    Packer packer{out.buffers, Pool::Weights, limits};
    const std::uint64_t limit = range_limit(limits);
    const auto tensors = index.tensors();
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const gguf::TensorEntry& t = tensors[i];
        const gguf::TensorShape shape{t.dimension_count,
                                      {t.dimensions[0], t.dimensions[1], t.dimensions[2], t.dimensions[3]},
                                      t.element_count};
        const bool tied = is_tied_copy(index, model, i);
        std::vector<WeightPiece> pieces;

        // A row is the first dimension: whole blocks, never split.
        const std::uint64_t rows = t.dimensions[0] == 0 ? 0 : t.element_count / t.dimensions[0];
        if (rows != 0) {
            const std::uint64_t row_bytes = t.data_length / rows;
            if (row_bytes > limit) return failure(PlanError::RowExceedsBinding, t.name);
            const std::uint64_t rows_per_piece = limit / row_bytes;
            for (std::uint64_t first = 0; first < rows; first += rows_per_piece) {
                const std::uint64_t count = std::min(rows_per_piece, rows - first);
                BufferRange range{};
                if (!packer.place(count * row_bytes, tied, range)) return failure(PlanError::Overflow, t.name);
                pieces.push_back({range.buffer, range.offset, range.length, first, count});
            }
        }
        out.tensors.push_back({static_cast<gguf::TensorId>(i), WeightView{t.type, shape, std::move(pieces)},
                               tied ? std::optional{model.token_embedding} : std::nullopt});
    }
    return {};
}

// The working buffers one layer's step needs, shared by every layer.
PlanResult place_scratch(const model::ModelDescription& model, const DeviceLimits& limits,
                         ResidencyPlan& out) {
    std::uint64_t query = 0, key_value = 0, feed_forward = 0;
    for (const model::LayerDescription& l : model.layers) {
        query = std::max<std::uint64_t>(query, std::uint64_t{l.query_heads} * l.head_dimension);
        key_value = std::max<std::uint64_t>(key_value, std::uint64_t{l.key_value_heads} * l.head_dimension);
        feed_forward = std::max<std::uint64_t>(feed_forward, l.feed_forward_width);
    }
    struct Need {
        std::string_view purpose;
        std::uint64_t rows;
        std::uint64_t width;
    };
    const std::array needs{
        Need{"hidden", kPrefillBlock, model.embedding_width},
        Need{"normed", kPrefillBlock, model.embedding_width},
        Need{"query", kPrefillBlock, query},
        Need{"key", kPrefillBlock, key_value},
        Need{"value", kPrefillBlock, key_value},
        Need{"attention", kPrefillBlock, query},
        Need{"gate", kPrefillBlock, feed_forward},
        Need{"up", kPrefillBlock, feed_forward},
        // Logits are computed for the last token of a step only.
        Need{"logits", 1, model.vocabulary_size},
    };
    Packer packer{out.buffers, Pool::Scratch, limits};
    for (const Need& need : needs) {
        std::uint64_t bytes = 0;
        if (!checked_mul(need.rows * need.width, kF32Bytes, bytes)) return failure(PlanError::Overflow, std::string(need.purpose));
        if (bytes > range_limit(limits)) return failure(PlanError::ScratchExceedsBinding, std::string(need.purpose));
        BufferRange range{};
        if (!packer.place(bytes, false, range)) return failure(PlanError::Overflow, std::string(need.purpose));
        out.scratch.push_back({need.purpose, range});
    }
    return {};
}

std::uint64_t pool_bytes(const std::vector<PlannedBuffer>& buffers, Pool pool) {
    std::uint64_t total = 0;
    for (const PlannedBuffer& b : buffers) {
        if (b.pool == pool) total += b.size;
    }
    return total;
}

gguf::TensorType storage_type(policy::CachePrecision precision) {
    switch (precision) {
        case policy::CachePrecision::F16: return gguf::TensorType::F16;
        case policy::CachePrecision::BF16: return gguf::TensorType::BF16;
        case policy::CachePrecision::Q8_0: return gguf::TensorType::Q8_0;
    }
    return gguf::TensorType::F16;
}

// Sizes the cache to the largest context the budget and the binding limit
// allow, and places it.
PlanResult place_cache(const model::ModelDescription& model, const DeviceLimits& limits,
                       const policy::LoadPolicy& policy, ResidencyPlan& out) {
    const gguf::FormatLayout& layout = *gguf::format_layout(storage_type(policy.cache_precision));
    const std::uint64_t limit = range_limit(limits);

    // Bytes one token adds to one layer's keys (and as many to its values).
    std::vector<std::uint64_t> per_token;
    std::uint64_t all_layers = 0;
    std::uint64_t binding_context = model.trained_context;
    for (std::size_t i = 0; i < model.layers.size(); ++i) {
        const model::LayerDescription& l = model.layers[i];
        if (l.head_dimension % layout.block_elements != 0) {
            return failure(PlanError::UnsupportedCachePrecision, "layer " + std::to_string(i));
        }
        const std::uint64_t bytes =
            std::uint64_t{l.key_value_heads} * (l.head_dimension / layout.block_elements) * layout.block_bytes;
        per_token.push_back(bytes);
        all_layers += 2 * bytes;
        binding_context = std::min(binding_context, limit / bytes);
    }
    // The binding never limits the cache below a prefill block: the "key"
    // working buffer holds kPrefillBlock tokens of the same width at 4 bytes a
    // value and already fits one binding, and every cache precision stores
    // fewer bytes a value than that.
    const std::uint64_t shortest = std::min<std::uint64_t>(kPrefillBlock, model.trained_context);

    // Alignment padding the cache may add: at most one granule per range.
    const std::uint64_t slack = 2 * model.layers.size() * (limits.storage_offset_alignment + kBindingGranule);
    const std::uint64_t fixed = out.weight_bytes + out.scratch_bytes + slack;
    const std::uint64_t budget = policy.memory_budget;
    const std::uint64_t budget_context = budget > fixed ? (budget - fixed) / all_layers : 0;
    if (budget_context < shortest) {
        out.total_bytes = fixed + shortest * all_layers;
        return failure(PlanError::ExceedsBudget);
    }
    out.context_offered = static_cast<std::uint32_t>(std::min(binding_context, budget_context));

    Packer packer{out.buffers, Pool::Cache, limits};
    for (const std::uint64_t bytes : per_token) {
        PlannedCacheLayer layer{};
        const std::uint64_t length = out.context_offered * bytes;
        if (!packer.place(length, false, layer.keys) || !packer.place(length, false, layer.values)) {
            return failure(PlanError::Overflow);
        }
        out.cache.push_back(layer);
    }
    return {};
}

}  // namespace

PlanResult plan_residency(const gguf::TensorIndex& index, const model::ModelDescription& model,
                          const DeviceLimits& limits, const policy::LoadPolicy& policy,
                          ResidencyPlan& out) {
    out = ResidencyPlan{};
    if (auto r = place_weights(index, model, limits, out); !r.ok()) return r;
    out.weight_bytes = pool_bytes(out.buffers, Pool::Weights);
    if (auto r = place_scratch(model, limits, out); !r.ok()) return r;
    out.scratch_bytes = pool_bytes(out.buffers, Pool::Scratch);
    if (auto r = place_cache(model, limits, policy, out); !r.ok()) return r;
    out.cache_bytes = pool_bytes(out.buffers, Pool::Cache);
    out.total_bytes = out.weight_bytes + out.scratch_bytes + out.cache_bytes;
    return {};
}

}  // namespace bllm::residency

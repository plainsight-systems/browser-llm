#include <doctest/doctest.h>

#include <algorithm>
#include <set>
#include <string>

#include "core/arch/architecture.h"
#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/residency/plan.h"
#include "support/gguf_fixture.h"

using namespace bllm;
using residency::DeviceLimits;
using residency::PlanError;
using residency::Pool;
using residency::ResidencyPlan;

namespace {

// A tiny model, read and described, ready to plan. The bytes stay alive with
// the index that refers to them.
struct Planned {
    std::vector<std::byte> bytes;
    gguf::TensorIndex index;
    model::ModelDescription model;
};

Planned describe(const std::string& fixture) {
    Planned p;
    p.bytes = testing::load_gguf_fixture(fixture);
    gguf::MemoryByteSource source{p.bytes};
    REQUIRE(gguf::read_index(source, p.index).error == gguf::ReadError::Ok);
    std::string_view name;
    REQUIRE(p.index.read_string("general.architecture", name) == gguf::MetadataError::Ok);
    REQUIRE(capability::find_architecture(name)->describe(p.index, p.model).ok());
    return p;
}

// WebGPU's defaults.
constexpr DeviceLimits kDefaults{256ull << 20, 128ull << 20, 256};

// The invariants every plan keeps, whatever the limits.
void check_invariants(const ResidencyPlan& plan, const DeviceLimits& limits) {
    for (const auto& b : plan.buffers) CHECK(b.size <= limits.max_buffer_size);
    auto check_range = [&](const residency::BufferRange& r, Pool pool) {
        REQUIRE(static_cast<std::size_t>(r.buffer) < plan.buffers.size());
        const auto& buffer = plan.buffers[static_cast<std::size_t>(r.buffer)];
        CHECK(buffer.pool == pool);
        CHECK(r.offset % limits.storage_offset_alignment == 0);
        CHECK(r.length <= limits.max_storage_binding_size);
        CHECK(r.offset + r.length <= buffer.size);
    };
    for (const auto& t : plan.tensors) {
        std::uint64_t next_row = 0;
        for (const auto& piece : t.view.pieces()) {
            check_range({piece.buffer, piece.offset, piece.length}, Pool::Weights);
            CHECK(piece.first_row == next_row);   // pieces cover the rows, in order
            next_row += piece.row_count;
        }
    }
    for (const auto& layer : plan.cache) {
        check_range(layer.keys, Pool::Cache);
        check_range(layer.values, Pool::Cache);
    }
    for (const auto& s : plan.scratch) check_range(s.range, Pool::Scratch);
    CHECK(plan.total_bytes == plan.weight_bytes + plan.cache_bytes + plan.scratch_bytes);
}

}  // namespace

TEST_CASE("every tensor is placed, whole, and within the limits") {
    const auto p = describe("tiny_qwen3");
    ResidencyPlan plan;
    REQUIRE(residency::plan_residency(p.index, p.model, kDefaults, policy::LoadPolicy{}, plan).ok());
    check_invariants(plan, kDefaults);

    REQUIRE(plan.tensors.size() == p.index.tensors().size());
    for (std::size_t i = 0; i < plan.tensors.size(); ++i) {
        const auto& entry = p.index.tensors()[i];
        std::uint64_t bytes = 0;
        for (const auto& piece : plan.tensors[i].view.pieces()) bytes += piece.length;
        CHECK(bytes == entry.data_length);
        CHECK(plan.tensors[i].view.format() == entry.type);
    }
    // A model this small packs into one weight buffer.
    CHECK(std::count_if(plan.buffers.begin(), plan.buffers.end(),
                        [](const auto& b) { return b.pool == Pool::Weights; }) == 1);
}

TEST_CASE("a weight wider than a binding is split by whole rows") {
    const auto p = describe("tiny_qwen3");
    // token_embd is 6 rows of 32 bytes: a 64-byte binding takes two rows.
    const DeviceLimits narrow{4096, 64, 32};
    // Room for the working buffers is beside the point here: give them none.
    ResidencyPlan plan;
    const auto r = residency::plan_residency(p.index, p.model, narrow, policy::LoadPolicy{}, plan);
    CHECK(r.error == PlanError::ScratchExceedsBinding);   // weights placed first
    const auto embd = static_cast<std::size_t>(*p.index.find("token_embd.weight"));
    REQUIRE(plan.tensors.size() > embd);
    const auto& pieces = plan.tensors[embd].view.pieces();
    REQUIRE(pieces.size() == 3);
    for (const auto& piece : pieces) {
        CHECK(piece.row_count == 2);
        CHECK(piece.length == 64);
        CHECK(piece.offset % narrow.storage_offset_alignment == 0);
    }
}

TEST_CASE("a row wider than a binding cannot be placed, and is named") {
    const auto p = describe("tiny_qwen3");
    ResidencyPlan plan;
    const auto r = residency::plan_residency(p.index, p.model, DeviceLimits{4096, 16, 32},
                                             policy::LoadPolicy{}, plan);
    CHECK(r.error == PlanError::RowExceedsBinding);
    CHECK(r.subject == "token_embd.weight");
}

TEST_CASE("a working buffer wider than a binding is named") {
    const auto p = describe("tiny_qwen3");
    ResidencyPlan plan;
    // Weights fit 1 KiB bindings; the 512-token hidden buffer (16 KiB) does not.
    const auto r = residency::plan_residency(p.index, p.model, DeviceLimits{1 << 20, 1024, 32},
                                             policy::LoadPolicy{}, plan);
    CHECK(r.error == PlanError::ScratchExceedsBinding);
    CHECK(r.subject == "hidden");
}

TEST_CASE("the context offered is capped by what the model was trained for") {
    const auto p = describe("tiny_qwen3");
    ResidencyPlan plan;
    REQUIRE(residency::plan_residency(p.index, p.model, kDefaults, policy::LoadPolicy{}, plan).ok());
    CHECK(plan.context_offered == p.model.trained_context);
    REQUIRE(plan.cache.size() == p.model.layers.size());
    // 1 key/value head of width 4 at f16: 8 bytes a token, for keys and for values.
    CHECK(plan.cache[0].keys.length == std::uint64_t{plan.context_offered} * 8);
}

TEST_CASE("the context offered is the most the budget allows, and fits within it") {
    auto p = describe("tiny_qwen3");
    // The fixture trains for 64 tokens, below a prefill block; a longer
    // trained context lets the budget, not the model, set the limit.
    p.model.trained_context = 4096;
    ResidencyPlan roomy;
    REQUIRE(residency::plan_residency(p.index, p.model, kDefaults, policy::LoadPolicy{}, roomy).ok());
    REQUIRE(roomy.context_offered == 4096);

    // 2 layers, keys and values, 8 bytes a token each: 32 bytes a token.
    // Take away 1,000 tokens' worth.
    policy::LoadPolicy tight{};
    tight.memory_budget = roomy.total_bytes - 32 * 1000;
    ResidencyPlan plan;
    REQUIRE(residency::plan_residency(p.index, p.model, kDefaults, tight, plan).ok());
    check_invariants(plan, kDefaults);
    CHECK(plan.total_bytes <= tight.memory_budget);
    CHECK(plan.context_offered <= 4096 - 1000);
    CHECK(plan.context_offered >= residency::kPrefillBlock);
}

TEST_CASE("a budget too small for the shortest context says what it needs") {
    const auto p = describe("tiny_qwen3");
    policy::LoadPolicy tiny{};
    tiny.memory_budget = 4096;
    ResidencyPlan plan;
    const auto r = residency::plan_residency(p.index, p.model, kDefaults, tiny, plan);
    CHECK(r.error == PlanError::ExceedsBudget);
    CHECK(plan.total_bytes > tiny.memory_budget);
}

TEST_CASE("a cache precision that cannot store the head dimension is named") {
    const auto p = describe("tiny_qwen3");
    policy::LoadPolicy q8{};
    q8.cache_precision = policy::CachePrecision::Q8_0;   // blocks of 32; heads are 4 wide
    ResidencyPlan plan;
    const auto r = residency::plan_residency(p.index, p.model, kDefaults, q8, plan);
    CHECK(r.error == PlanError::UnsupportedCachePrecision);
    CHECK(r.subject == "layer 0");
}

TEST_CASE("an output head stored as a copy of the embedding gets buffers of its own") {
    const auto p = describe("tiny_qwen3_output_copy");
    ResidencyPlan plan;
    REQUIRE(residency::plan_residency(p.index, p.model, kDefaults, policy::LoadPolicy{}, plan).ok());
    check_invariants(plan, kDefaults);

    const auto head = static_cast<std::size_t>(*p.model.output_head);
    REQUIRE(plan.tensors[head].candidate_duplicate_of == p.model.token_embedding);
    std::set<residency::BufferIndex> head_buffers;
    for (const auto& piece : plan.tensors[head].view.pieces()) head_buffers.insert(piece.buffer);
    for (std::size_t i = 0; i < plan.tensors.size(); ++i) {
        if (i == head) continue;
        for (const auto& piece : plan.tensors[i].view.pieces()) {
            CHECK(head_buffers.count(piece.buffer) == 0);   // nothing else shares them
        }
        CHECK_FALSE(plan.tensors[i].candidate_duplicate_of.has_value());
    }
}

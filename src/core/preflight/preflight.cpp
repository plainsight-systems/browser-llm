#include "core/preflight/preflight.h"

#include <algorithm>
#include <string_view>
#include <vector>

#include "core/arch/architecture.h"
#include "core/capability/capability.h"
#include "core/formats/format.h"
#include "core/model/model_description.h"
#include "core/tokenizer/tokenizer.h"

namespace bllm::preflight {
namespace {

std::string quoted(std::string_view text) {
    return "\"" + std::string(text) + "\"";
}

// Why a required string key could not be read, for a rejection message.
std::string unreadable_key(gguf::MetadataError error, std::string_view key) {
    return error == gguf::MetadataError::MissingKey
               ? "the file does not declare " + std::string(key)
               : std::string(key) + " is not a string";
}

std::string_view to_string(arch::DescribeError error) noexcept {
    switch (error) {
        case arch::DescribeError::Ok: return "ok";
        case arch::DescribeError::MissingKey: return "a required key is missing";
        case arch::DescribeError::WrongKeyType: return "a key has the wrong type";
        case arch::DescribeError::MissingTensor: return "a required tensor is missing";
        case arch::DescribeError::ShapeMismatch: return "a tensor has the wrong shape";
        case arch::DescribeError::InvalidValue: return "a value no model could have";
        case arch::DescribeError::UnsupportedValue: return "a value in a form this build does not read";
    }
    return "unrecognised error";
}

// Describe: the architecture is implemented and can read its numbers from
// this file. Returns the description, which Fit needs.
std::optional<model::ModelDescription> check_architecture(const gguf::TensorIndex& index,
                                                          Verdict& verdict) {
    constexpr std::string_view kKey = "general.architecture";
    std::string_view name;
    if (const auto e = index.read_string(kKey, name); e != gguf::MetadataError::Ok) {
        verdict.blockers.push_back({Stage::Describe, unreadable_key(e, kKey)});
        return std::nullopt;
    }
    const arch::Architecture* architecture = capability::find_architecture(name);
    if (architecture == nullptr) {
        verdict.blockers.push_back(
            {Stage::Describe, "architecture " + quoted(name) + " is not supported"});
        return std::nullopt;
    }
    model::ModelDescription description{};
    if (const auto r = architecture->describe(index, description); !r.ok()) {
        verdict.blockers.push_back(
            {Stage::Describe, "architecture " + quoted(name) + " cannot read this file: " +
                                  std::string(to_string(r.error)) + " (" + r.subject + ")"});
        return std::nullopt;
    }
    return description;
}

std::string mebibytes(std::uint64_t bytes) {
    return std::to_string((bytes + (1u << 19)) >> 20) + " MiB";
}

std::string fit_failure(const residency::PlanResult& r, const residency::ResidencyPlan& plan,
                        const policy::LoadPolicy& policy) {
    using residency::PlanError;
    switch (r.error) {
        case PlanError::Ok: return "fits";
        case PlanError::ExceedsBudget:
            return "needs " + mebibytes(plan.total_bytes) + " for the shortest context worth offering (" +
                   std::to_string(residency::kPrefillBlock) + " tokens); the memory budget is " +
                   mebibytes(policy.memory_budget);
        case PlanError::RowExceedsBinding:
            return "a row of " + r.subject + " is wider than one storage binding";
        case PlanError::ScratchExceedsBinding:
            return "the " + r.subject + " working buffer is wider than one storage binding";
        case PlanError::UnsupportedCachePrecision:
            return "the cache precision cannot store the head dimension of " + r.subject;
        case PlanError::Overflow:
            return "sizes overflow (" + r.subject + ")";
    }
    return "unrecognised error";
}

// Fit: the residency plan fits the granted limits and the memory budget.
void check_fit(const gguf::TensorIndex& index, const model::ModelDescription& description,
               const residency::DeviceLimits& limits, const policy::LoadPolicy& policy,
               Verdict& verdict) {
    if (limits.max_buffer_size == 0 || limits.max_storage_binding_size == 0 ||
        limits.storage_offset_alignment == 0) {
        verdict.blockers.push_back({Stage::Fit, "no GPU device was acquired, so fit cannot be judged"});
        return;
    }
    if ((limits.storage_offset_alignment & (limits.storage_offset_alignment - 1)) != 0) {
        verdict.blockers.push_back({Stage::Fit, "the device's storage-offset alignment is not a power of two"});
        return;
    }
    residency::ResidencyPlan plan;
    if (const auto r = residency::plan_residency(index, description, limits, policy, plan); !r.ok()) {
        verdict.blockers.push_back({Stage::Fit, fit_failure(r, plan, policy)});
        return;
    }
    verdict.fit = FitSummary{plan.weight_bytes,    plan.cache_bytes,       plan.scratch_bytes,
                             plan.total_bytes,     policy.memory_budget,   plan.context_offered,
                             description.trained_context, plan.buffers.size(), {}};
    for (const residency::PlannedTensor& t : plan.tensors) {
        if (!t.candidate_duplicate_of) continue;
        const gguf::TensorEntry& copy = index.tensor(t.tensor);
        const gguf::TensorEntry& original = index.tensor(*t.candidate_duplicate_of);
        verdict.fit->duplicates.push_back(
            {t.tensor, *t.candidate_duplicate_of, copy.data_offset, original.data_offset, copy.data_length});
    }
}

// Run needs every weight format. One blocker per unsupported format, naming
// how many tensors use it and the first of them.
void check_formats(const gguf::TensorIndex& index, Verdict& verdict) {
    struct Unsupported {
        gguf::TensorType type;
        std::size_t users;
        std::string_view first_user;
    };
    std::vector<Unsupported> found;
    for (const gguf::TensorEntry& tensor : index.tensors()) {
        if (capability::find_format(tensor.type) != nullptr) continue;
        auto it = std::find_if(found.begin(), found.end(),
                               [&](const Unsupported& u) { return u.type == tensor.type; });
        if (it == found.end()) {
            found.push_back({tensor.type, 1, tensor.name});
        } else {
            ++it->users;
        }
    }
    for (const Unsupported& u : found) {
        verdict.blockers.push_back(
            {Stage::Run, "format " + std::string(gguf::format_layout(u.type)->name) +
                                " is not supported (" + std::to_string(u.users) +
                                (u.users == 1 ? " tensor" : " tensors") + ", first " +
                                std::string(u.first_user) + ")"});
    }
}

// Run needs every row a whole number of unpack's groups (format.h), as
// routes does; one blocker, naming how many tensors fall short and the first.
void check_rows(const gguf::TensorIndex& index, Verdict& verdict) {
    std::size_t short_rows = 0;
    const gguf::TensorEntry* first = nullptr;
    for (const gguf::TensorEntry& tensor : index.tensors()) {
        if (formats::steps_by_groups(tensor)) continue;
        if (first == nullptr) first = &tensor;
        ++short_rows;
    }
    if (first == nullptr) return;
    verdict.blockers.push_back(
        {Stage::Run, "rows must be a multiple of " + std::to_string(formats::kUnpackGroup) + " weights (" +
                         std::to_string(short_rows) + (short_rows == 1 ? " tensor" : " tensors") +
                         ", first " + first->name + ", rows of " + std::to_string(first->dimensions[0]) + ")"});
}

// Run needs the tokenizer, and the pre-tokenizer when the tokenizer splits
// text first. One that splits none ignores tokenizer.ggml.pre, as llama.cpp
// does: its converter writes "default" there for SentencePiece files.
void check_tokenizer(const gguf::TensorIndex& index, Verdict& verdict) {
    constexpr std::string_view kModelKey = "tokenizer.ggml.model";
    constexpr std::string_view kPreKey = "tokenizer.ggml.pre";

    std::string_view model_name;
    const tokenizer::Algorithm* algorithm = nullptr;
    if (const auto e = index.read_string(kModelKey, model_name); e != gguf::MetadataError::Ok) {
        verdict.blockers.push_back({Stage::Run, unreadable_key(e, kModelKey)});
    } else if (algorithm = capability::find_tokenizer(model_name); algorithm == nullptr) {
        verdict.blockers.push_back(
            {Stage::Run, "tokenizer " + quoted(model_name) + " is not supported"});
    }

    if (algorithm != nullptr && !algorithm->requires_pretokenizer) return;

    std::string_view pre_name;
    const auto pre = index.read_string(kPreKey, pre_name);
    if (pre == gguf::MetadataError::Ok) {
        if (capability::find_pretokenizer(pre_name) == nullptr) {
            verdict.blockers.push_back(
                {Stage::Run, "pre-tokenizer " + quoted(pre_name) + " is not supported"});
        }
    } else if (algorithm != nullptr && algorithm->requires_pretokenizer) {
        verdict.blockers.push_back(
            {Stage::Run, "tokenizer " + quoted(model_name) +
                                  " needs a pre-tokenizer, and " + unreadable_key(pre, kPreKey)});
    }
}

}  // namespace

Verdict preflight(const gguf::TensorIndex& index, const residency::DeviceLimits& limits,
                  const policy::LoadPolicy& policy) {
    Verdict verdict;
    if (const auto description = check_architecture(index, verdict)) {
        check_fit(index, *description, limits, policy, verdict);
    }
    check_formats(index, verdict);
    check_rows(index, verdict);
    check_tokenizer(index, verdict);

    for (int s = static_cast<int>(kImplementedThrough) + 1;
         s <= static_cast<int>(Stage::Run); ++s) {
        const auto stage = static_cast<Stage>(s);
        verdict.blockers.push_back(
            {stage, "the " + std::string(to_string(stage)) + " stage is not implemented in this build"});
    }
    return verdict;
}

}  // namespace bllm::preflight

#include "core/preflight/preflight.h"

#include <algorithm>
#include <string_view>
#include <vector>

#include "core/arch/architecture.h"
#include "core/capability/capability.h"
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
// this file.
void check_architecture(const gguf::TensorIndex& index, Verdict& verdict) {
    constexpr std::string_view kKey = "general.architecture";
    std::string_view name;
    if (const auto e = index.read_string(kKey, name); e != gguf::MetadataError::Ok) {
        verdict.blockers.push_back({Stage::Describe, unreadable_key(e, kKey)});
        return;
    }
    const arch::Architecture* architecture = capability::find_architecture(name);
    if (architecture == nullptr) {
        verdict.blockers.push_back(
            {Stage::Describe, "architecture " + quoted(name) + " is not supported"});
        return;
    }
    model::ModelDescription description{};
    if (const auto r = architecture->describe(index, description); !r.ok()) {
        verdict.blockers.push_back(
            {Stage::Describe, "architecture " + quoted(name) + " cannot read this file: " +
                                  std::string(to_string(r.error)) + " (" + r.subject + ")"});
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

// Run needs the tokenizer, and the pre-tokenizer when the tokenizer splits
// text first.
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

Verdict preflight(const gguf::TensorIndex& index,
                  [[maybe_unused]] const residency::DeviceLimits& limits,
                  [[maybe_unused]] const policy::LoadPolicy& policy) {
    Verdict verdict;
    check_architecture(index, verdict);
    check_formats(index, verdict);
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

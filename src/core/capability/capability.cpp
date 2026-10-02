#include "core/capability/capability.h"

#include <array>
#include <span>

#include "core/arch/gemma3/gemma3.h"
#include "core/arch/llama/llama.h"
#include "core/arch/qwen3/qwen3.h"
#include "core/tokenizer/bpe/byte_level_bpe.h"
#include "core/tokenizer/bpe/sentencepiece_bpe.h"
#include "core/tokenizer/pretokenize.h"

namespace bllm::capability {
namespace {

// One identifier a file can carry, and the implementation that runs it.
// Several rows may share an implementation (principle 2).
template <typename Identifier, typename Implementation>
struct Row {
    Identifier identifier;
    const Implementation* implementation;
};

template <typename Identifier, typename Implementation>
const Implementation* lookup(std::span<const Row<Identifier, Implementation>> table,
                             Identifier identifier) noexcept {
    for (const auto& row : table) {
        if (row.identifier == identifier) return row.implementation;
    }
    return nullptr;
}

// An identifier is supported exactly when it has a row here. Adding an
// implementation adds its file and one row, and nothing else.
constexpr std::array kArchitectures{
    Row<std::string_view, arch::Architecture>{"gemma3", &arch::kGemma3},
    Row<std::string_view, arch::Architecture>{"llama", &arch::kLlama},
    Row<std::string_view, arch::Architecture>{"qwen3", &arch::kQwen3},
};
constexpr std::array<Row<gguf::TensorType, formats::Format>, 0> kFormats{};
constexpr std::array kTokenizers{
    Row<std::string_view, tokenizer::Algorithm>{"gpt2", &tokenizer::bpe::kByteLevel},
    Row<std::string_view, tokenizer::Algorithm>{"llama", &tokenizer::bpe::kSentencePiece},
};
constexpr std::array kPreTokenizers{
    Row<std::string_view, tokenizer::PreTokenizer>{"llama-bpe", &tokenizer::kLlamaBpe},
    Row<std::string_view, tokenizer::PreTokenizer>{"qwen2", &tokenizer::kQwen2},
};

}  // namespace

const arch::Architecture* find_architecture(std::string_view general_architecture) noexcept {
    return lookup<std::string_view, arch::Architecture>(kArchitectures, general_architecture);
}

const formats::Format* find_format(gguf::TensorType type) noexcept {
    return lookup<gguf::TensorType, formats::Format>(kFormats, type);
}

const tokenizer::Algorithm* find_tokenizer(std::string_view tokenizer_model) noexcept {
    return lookup<std::string_view, tokenizer::Algorithm>(kTokenizers, tokenizer_model);
}

const tokenizer::PreTokenizer* find_pretokenizer(std::string_view tokenizer_pre) noexcept {
    return lookup<std::string_view, tokenizer::PreTokenizer>(kPreTokenizers, tokenizer_pre);
}

}  // namespace bllm::capability

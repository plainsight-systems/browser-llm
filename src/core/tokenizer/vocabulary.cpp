#include "core/tokenizer/vocabulary.h"

#include <algorithm>
#include <string>
#include <utility>

#include "core/tokenizer/load.h"

namespace bllm::tokenizer {

std::optional<TokenId> Vocabulary::find(std::string_view text) const noexcept {
    const auto at = std::lower_bound(by_text_.begin(), by_text_.end(), text,
                                     [&](TokenId id, std::string_view t) { return this->text(id) < t; });
    if (at == by_text_.end() || this->text(*at) != text) return std::nullopt;
    return *at;
}

LoadResult load_vocabulary(gguf::ByteSource& source, const gguf::TensorIndex& index, Vocabulary& out) {
    constexpr std::string_view kTokens = "tokenizer.ggml.tokens";
    constexpr std::string_view kTypes = "tokenizer.ggml.token_type";

    Vocabulary vocabulary;
    if (auto r = read_string_array(source, index, kTokens, vocabulary.tokens_); !r.ok()) return r;
    std::vector<std::int32_t> types;
    if (auto r = read_int32_array(source, index, kTypes, types); !r.ok()) return r;
    if (types.size() != vocabulary.tokens_.size()) return {LoadError::CountMismatch, std::string(kTypes)};

    vocabulary.types_.reserve(types.size());
    for (std::size_t i = 0; i < types.size(); ++i) {
        if (types[i] < 0 || types[i] > static_cast<std::int32_t>(TokenType::Byte)) {
            return {LoadError::UnknownTokenType, "token " + std::to_string(i)};
        }
        vocabulary.types_.push_back(static_cast<TokenType>(types[i]));
    }

    vocabulary.by_text_.resize(types.size());
    for (std::size_t i = 0; i < types.size(); ++i) vocabulary.by_text_[i] = static_cast<TokenId>(i);
    const auto by_text = [&](TokenId a, TokenId b) { return vocabulary.text(a) < vocabulary.text(b); };
    std::sort(vocabulary.by_text_.begin(), vocabulary.by_text_.end(), by_text);
    const auto repeat = std::adjacent_find(vocabulary.by_text_.begin(), vocabulary.by_text_.end(),
                                           [&](TokenId a, TokenId b) { return vocabulary.text(a) == vocabulary.text(b); });
    if (repeat != vocabulary.by_text_.end()) return {LoadError::DuplicateToken, std::string(vocabulary.text(*repeat))};

    out = std::move(vocabulary);
    return {};
}

}  // namespace bllm::tokenizer

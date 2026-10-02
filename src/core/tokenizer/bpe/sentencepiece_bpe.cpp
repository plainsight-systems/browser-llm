#include "core/tokenizer/bpe/sentencepiece_bpe.h"

#include <cstdio>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "core/tokenizer/bpe/merge.h"
#include "core/tokenizer/bpe/sentencepiece_merges.h"
#include "core/tokenizer/load.h"
#include "core/tokenizer/unicode.h"

namespace bllm::tokenizer::bpe {
namespace {

constexpr std::string_view kSpace = "\xE2\x96\x81";   // U+2581, how the vocabulary spells a space

// Whether `text` is exactly one character.
bool one_character(std::string_view text) noexcept {
    Utf8Char c{};
    return !text.empty() && decode_utf8(text, 0, c) && c.length == text.size();
}

}  // namespace

EncodeError SentencePieceBpe::encode(std::string_view text, std::vector<TokenId>& out) const {
    if (text.size() > std::numeric_limits<std::uint32_t>::max()) return EncodeError::TooLong;

    std::vector<Segment> segments;
    special_.segment(text, segments);

    std::vector<TokenId> tokens;
    std::string spelled;   // scratch for each character
    std::vector<TokenId> symbols;
    for (const Segment& segment : segments) {
        if (segment.special) {
            tokens.push_back(*segment.special);
            continue;
        }
        const std::string_view ordinary = text.substr(segment.offset, segment.length);
        symbols.clear();
        for (std::size_t at = 0; at < ordinary.size();) {
            Utf8Char c{};
            if (!decode_utf8(ordinary, at, c)) return EncodeError::InvalidUtf8;
            const std::string_view character = ordinary.substr(at, c.length);
            spelled.assign(c.code_point == U' ' ? kSpace : character);
            const auto token = vocabulary_.find(spelled);
            if (token && vocabulary_.type(*token) == TokenType::Normal) {
                symbols.push_back(*token);
            } else {
                for (const char byte : spelled) symbols.push_back(byte_tokens_[static_cast<unsigned char>(byte)]);
            }
            at += c.length;
        }
        merge(symbols, merges_, tokens);
    }
    out.insert(out.end(), tokens.begin(), tokens.end());
    return EncodeError::Ok;
}

LoadResult load_sentencepiece_bpe(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                  SentencePieceBpe& out) {
    constexpr std::string_view kSpacePrefix = "tokenizer.ggml.add_space_prefix";
    bool space_prefix = true;
    switch (index.read_bool(kSpacePrefix, space_prefix)) {
        case gguf::MetadataError::Ok: break;
        case gguf::MetadataError::MissingKey: return {LoadError::Unsupported, std::string(kSpacePrefix) + " is not declared"};
        case gguf::MetadataError::WrongType: return {LoadError::WrongKeyType, std::string(kSpacePrefix)};
    }
    if (space_prefix) return {LoadError::Unsupported, std::string(kSpacePrefix) + " is true"};

    SentencePieceBpe spm;
    if (auto r = load_vocabulary(source, index, spm.vocabulary_); !r.ok()) return r;
    std::vector<float> scores;
    if (auto r = read_float32_array(source, index, "tokenizer.ggml.scores", scores); !r.ok()) return r;
    if (scores.size() != spm.vocabulary_.size()) return {LoadError::CountMismatch, "tokenizer.ggml.scores"};
    if (auto r = implied_merges(spm.vocabulary_, scores, spm.merges_); !r.ok()) return r;

    // A character becomes its normal token, and a special token's text is
    // matched before that. One with a token of another type (unused, unknown)
    // would be that token to both references and its bytes here: refused.
    for (std::size_t id = 0; id < spm.vocabulary_.size(); ++id) {
        const auto token = static_cast<TokenId>(id);
        const TokenType type = spm.vocabulary_.type(token);
        if (type == TokenType::Normal || type == TokenType::Control || type == TokenType::UserDefined ||
            type == TokenType::Byte || !one_character(spm.vocabulary_.text(token))) {
            continue;
        }
        return {LoadError::Unsupported, "token " + std::to_string(id) + ", one character that is not a normal token"};
    }

    for (std::size_t byte = 0; byte < 256; ++byte) {
        char name[8];
        std::snprintf(name, sizeof name, "<0x%02zX>", byte);
        const auto token = spm.vocabulary_.find(name);
        if (!token || spm.vocabulary_.type(*token) != TokenType::Byte) {
            return {LoadError::MissingByte, "byte " + std::string(name)};
        }
        spm.byte_tokens_[byte] = *token;
    }
    spm.special_ = SpecialTokens{spm.vocabulary_};
    out = std::move(spm);
    return {};
}

}  // namespace bllm::tokenizer::bpe

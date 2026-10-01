#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/arrays.h"
#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/tokenizer/tokenizer.h"

namespace bllm::tokenizer {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// A model's vocabulary, read from tokenizer.ggml.tokens and
// tokenizer.ggml.token_type: every token's text and type, by identifier, and
// the identifier of a given text. Shared by every algorithm.
//
// A token's text is unique. No vocabulary listed here repeats one, and the
// references disagree on which repeat would win, so a file that repeats one is
// refused rather than read by a guess.

// Token types, numbered as GGUF numbers them.
enum class TokenType : std::int32_t {
    Undefined = 0,
    Normal = 1,
    Unknown = 2,
    Control = 3,       // special: written as text by the chat template, never merged into
    UserDefined = 4,   // special: matched in the raw text as it stands
    Unused = 5,
    Byte = 6,          // one byte, for algorithms that fall back to bytes
};

enum class LoadError {
    Ok,
    MissingKey,        // the subject names it
    WrongKeyType,      // the subject names it
    Unreadable,        // the array could not be read; the subject says why
    CountMismatch,     // the arrays disagree on how many tokens there are
    UnknownTokenType,  // the subject names the token
    DuplicateToken,    // the subject names the text
    MalformedMerge,    // not two tokens separated by one space; the subject names it
    UnknownMergeToken, // a merge names a token the vocabulary lacks
    DuplicateMerge,
    MissingByte,       // a byte-level vocabulary lacks a byte's token
};

struct LoadResult {
    LoadError error = LoadError::Ok;
    std::string subject;

    [[nodiscard]] bool ok() const noexcept { return error == LoadError::Ok; }
};

class Vocabulary {
public:
    [[nodiscard]] std::size_t size() const noexcept { return types_.size(); }

    // Precondition: the identifier is below size().
    [[nodiscard]] std::string_view text(TokenId id) const noexcept {
        return tokens_[static_cast<std::size_t>(id)];
    }
    [[nodiscard]] TokenType type(TokenId id) const noexcept { return types_[static_cast<std::size_t>(id)]; }

    // The token whose text is exactly `text`, if there is one.
    [[nodiscard]] std::optional<TokenId> find(std::string_view text) const noexcept;

private:
    friend LoadResult load_vocabulary(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                      Vocabulary& out);

    gguf::StringTable tokens_;
    std::vector<TokenType> types_;
    std::vector<TokenId> by_text_;   // every identifier, ordered by its text
};

// Reads the vocabulary. `out` is left untouched unless the read succeeds.
[[nodiscard]] LoadResult load_vocabulary(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                         Vocabulary& out);

}  // namespace bllm::tokenizer

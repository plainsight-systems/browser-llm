#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace bllm::tokenizer {

// Contract 9: the tokenizer.
//
// Each tokenizer/<algorithm>/ provides one Algorithm, and the capability table
// lists it under the tokenizer.ggml.model value it implements. It builds its
// tokenizer from the vocabulary arrays the tensor index locates.
//
//   - Encoding turns rendered text into identifiers. The chat template writes
//     special tokens as text; they encode as their single identifiers, never
//     as the characters that spell them.
//   - An algorithm that splits text first names its pre-tokenizer with
//     tokenizer.ggml.pre. One that does not, such as SentencePiece, needs none.
//   - Decoding is a stream (Utf8Stream below).

// A token's identifier. Its own type, so it cannot be passed where a count or
// a position is meant.
enum class TokenId : std::uint32_t {};

struct Algorithm {
    std::string_view name;
    bool requires_pretokenizer;
};

enum class EncodeError {
    Ok,
    InvalidUtf8,
};

// Turns the bytes of successive tokens into text. A token can end partway
// through a UTF-8 character; those bytes are held until the next token
// completes them, so every string emitted is whole characters. Bytes that
// cannot begin or continue a character are replaced with U+FFFD.
class Utf8Stream {
public:
    // Appends every character completed by `token_bytes` to `out`.
    void push(std::string_view token_bytes, std::string& out);

    // Ends the stream. Returns false if bytes were pending — generation
    // stopped inside a character — in which case U+FFFD is appended in their
    // place rather than dropping them unseen.
    [[nodiscard]] bool finish(std::string& out);

private:
    std::array<char, 3> pending_{};
    std::uint8_t pending_count_ = 0;
};

}  // namespace bllm::tokenizer

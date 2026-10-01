#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace bllm::tokenizer {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// Unicode as tokenization needs it: UTF-8 decoding, and the character classes
// the split patterns test. The classes come from tables generated from one
// version of the Unicode Character Database (unicode_tables.h) and compiled
// in, so every browser classifies the same text the same way.

// A code point's class as the split patterns test it. A code point is in at
// most one of these: no letter or number is white space.
enum class CharClass : std::uint8_t {
    Other,
    Letter,       // \p{L}: general category L*
    Number,       // \p{N}: general category N*
    Whitespace,   // \s: the White_Space property
};

[[nodiscard]] CharClass char_class(char32_t code_point) noexcept;

// One character decoded from UTF-8.
struct Utf8Char {
    char32_t code_point;
    std::uint8_t length;   // bytes it took, 1 to 4
};

// Decodes the character that starts at text[at]. False when the bytes there
// are not well-formed UTF-8: a stray continuation byte, a truncated sequence,
// an overlong encoding, a surrogate, or a value past U+10FFFF.
// Precondition: at < text.size().
[[nodiscard]] bool decode_utf8(std::string_view text, std::size_t at, Utf8Char& out) noexcept;

// Appends the UTF-8 encoding of `code_point`. Precondition: it is a Unicode
// scalar value (at most U+10FFFF, not a surrogate).
void append_utf8(char32_t code_point, std::string& out);

}  // namespace bllm::tokenizer

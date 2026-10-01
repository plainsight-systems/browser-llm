#pragma once

#include <array>
#include <cstdint>
#include <optional>

namespace bllm::tokenizer::bpe {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// Byte-level BPE spells every byte as one printable character, so a
// vocabulary of text strings can hold any bytes at all. GPT-2 defined the
// mapping and every byte-level vocabulary since uses it: the 188 bytes that
// print as themselves ('!'..'~', '¡'..'¬', '®'..'ÿ') stand for themselves,
// and the other 68, in byte order, become U+0100 onwards. So a space is "Ġ"
// (U+0120) and a newline "Ċ" (U+010A).

namespace detail {

constexpr bool prints_as_itself(std::uint32_t byte) {
    return (byte >= 0x21 && byte <= 0x7E) || (byte >= 0xA1 && byte <= 0xAC) || (byte >= 0xAE && byte <= 0xFF);
}

consteval std::array<char32_t, 256> make_byte_chars() {
    std::array<char32_t, 256> chars{};
    char32_t next = 0x100;
    for (std::uint32_t byte = 0; byte < 256; ++byte) chars[byte] = prints_as_itself(byte) ? byte : next++;
    return chars;
}

}  // namespace detail

// The character that stands for each byte.
inline constexpr std::array<char32_t, 256> kByteChars = detail::make_byte_chars();

// The byte a character stands for, if it stands for one.
[[nodiscard]] constexpr std::optional<std::uint8_t> byte_of(char32_t c) noexcept {
    if (c < 0x100) {
        if (detail::prints_as_itself(c)) return static_cast<std::uint8_t>(c);
        return std::nullopt;
    }
    for (std::uint32_t byte = 0; byte < 256; ++byte) {
        if (kByteChars[byte] == c) return static_cast<std::uint8_t>(byte);
    }
    return std::nullopt;
}

static_assert(kByteChars[' '] == 0x120 && kByteChars['\n'] == 0x10A && kByteChars['A'] == U'A');
static_assert(kByteChars[0xFF] == 0xFF && kByteChars[0xAD] == 0x143);   // the last of the 68 moved

}  // namespace bllm::tokenizer::bpe

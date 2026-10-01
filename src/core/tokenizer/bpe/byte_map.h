#pragma once

#include <array>
#include <cstddef>
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

// Every character the map uses is below U+0144: the 68 moved bytes end there.
inline constexpr std::size_t kCharLimit = 0x100 + 68;

// The byte each character below kCharLimit stands for, or -1.
consteval std::array<std::int16_t, kCharLimit> make_char_bytes() {
    std::array<std::int16_t, kCharLimit> bytes{};
    bytes.fill(-1);
    const std::array<char32_t, 256> chars = make_byte_chars();
    for (std::uint32_t byte = 0; byte < 256; ++byte) bytes[chars[byte]] = static_cast<std::int16_t>(byte);
    return bytes;
}

inline constexpr std::array<std::int16_t, kCharLimit> kCharBytes = make_char_bytes();

}  // namespace detail

// The character that stands for each byte.
inline constexpr std::array<char32_t, 256> kByteChars = detail::make_byte_chars();

// The byte a character stands for, if it stands for one.
//
// Optimization (practice): an index into the map inverted at compile time
// (EMB.6), where a search of kByteChars took up to 256 comparisons. Loading
// asks it of every character of every normal token: searching, that check
// added 10 ms to Qwen3's load; indexing, 3 ms.
[[nodiscard]] constexpr std::optional<std::uint8_t> byte_of(char32_t c) noexcept {
    if (c >= detail::kCharBytes.size() || detail::kCharBytes[c] < 0) return std::nullopt;
    return static_cast<std::uint8_t>(detail::kCharBytes[c]);
}

static_assert(kByteChars[' '] == 0x120 && kByteChars['\n'] == 0x10A && kByteChars['A'] == U'A');
static_assert(kByteChars[0xFF] == 0xFF && kByteChars[0xAD] == 0x143);   // the last of the 68 moved

}  // namespace bllm::tokenizer::bpe

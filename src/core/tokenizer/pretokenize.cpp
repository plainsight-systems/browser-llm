#include "core/tokenizer/pretokenize.h"

#include <array>
#include <limits>
#include <span>
#include <string_view>
#include <utility>

#include "core/tokenizer/unicode.h"

namespace bllm::tokenizer {

const PreTokenizer kQwen2{"qwen2", 1, Normalization::Nfc, false};
const PreTokenizer kLlamaBpe{"llama-bpe", 3, Normalization::None, true};

namespace {

// One decoded character, with what the pattern asks of it.
struct Char {
    char32_t code_point;
    CharClass char_class;
    std::uint32_t offset;   // where its bytes start in the text
};

using Text = std::span<const Char>;

bool line_end(char32_t c) { return c == U'\r' || c == U'\n'; }
bool letter(const Char& c) { return c.char_class == CharClass::Letter; }
bool number(const Char& c) { return c.char_class == CharClass::Number; }
bool space(const Char& c) { return c.char_class == CharClass::Whitespace; }
bool symbol(const Char& c) { return c.char_class == CharClass::Other; }   // [^\s\p{L}\p{N}]

// The first position at or after `i` that does not satisfy `test`.
template <typename Test>
std::size_t run_end(Text text, std::size_t i, Test test) {
    while (i < text.size() && test(text[i])) ++i;
    return i;
}

// A character under simple case folding, for the contraction letters. In
// CaseFolding.txt for Unicode 16.0 the only characters that fold to one of
// s t r e v m l d are their ASCII capitals and U+017F, long s, which folds to s.
char32_t fold(char32_t c) {
    if (c >= U'A' && c <= U'Z') return c - U'A' + U'a';
    if (c == 0x017F) return U's';
    return c;
}

// (?i:'s|'t|'re|'ve|'m|'ll|'d)
std::size_t contraction(Text text, std::size_t i) {
    if (text[i].code_point != U'\'') return 0;
    constexpr std::array<std::u32string_view, 7> kEndings{U"s", U"t", U"re", U"ve", U"m", U"ll", U"d"};
    for (const std::u32string_view ending : kEndings) {
        if (text.size() - i - 1 < ending.size()) continue;
        bool matches = true;
        for (std::size_t k = 0; k < ending.size(); ++k) matches &= fold(text[i + 1 + k].code_point) == ending[k];
        if (matches) return 1 + ending.size();
    }
    return 0;
}

// [^\r\n\p{L}\p{N}]?\p{L}+
std::size_t letters(Text text, std::size_t i) {
    std::size_t start = i;
    if (!letter(text[i])) {
        // The optional character, taken only when letters follow it.
        if (line_end(text[i].code_point) || number(text[i])) return 0;
        if (i + 1 == text.size() || !letter(text[i + 1])) return 0;
        start = i + 1;
    }
    return run_end(text, start, letter) - i;
}

// \p{N}{1,digits}
std::size_t numbers(Text text, std::size_t i, std::size_t digits) {
    std::size_t end = i;
    while (end < text.size() && end - i < digits && number(text[end])) ++end;
    return end - i;
}

//  ?[^\s\p{L}\p{N}]+[\r\n]*
std::size_t symbols(Text text, std::size_t i) {
    std::size_t start = i;
    // The optional space, taken only when a symbol follows it.
    if (text[i].code_point == U' ' && i + 1 < text.size() && symbol(text[i + 1])) start = i + 1;
    const std::size_t end = run_end(text, start, symbol);
    if (end == start) return 0;
    return run_end(text, end, [](const Char& c) { return line_end(c.code_point); }) - i;
}

// \s*[\r\n]+ : greedy white space backs off to the run's last line end, which
// the line ends then match.
std::size_t white_space_through_line_end(Text text, std::size_t i) {
    const std::size_t end = run_end(text, i, space);
    for (std::size_t k = end; k > i; --k) {
        if (line_end(text[k - 1].code_point)) return k - i;
    }
    return 0;
}

// \s+(?!\S) : a white space run that ends the text, or, when text follows,
// all of it but the last character, which then leads the next piece.
std::size_t white_space_not_before_text(Text text, std::size_t i) {
    const std::size_t end = run_end(text, i, space);
    if (end == text.size()) return end - i;
    return end - i >= 2 ? end - i - 1 : 0;
}

// \s+
std::size_t white_space(Text text, std::size_t i) {
    return run_end(text, i, space) - i;
}

// The length of the piece at `i`: the first alternative that matches.
std::size_t piece_at(Text text, std::size_t i, std::size_t digits) {
    if (const std::size_t n = contraction(text, i)) return n;
    if (const std::size_t n = letters(text, i)) return n;
    if (const std::size_t n = numbers(text, i, digits)) return n;
    if (const std::size_t n = symbols(text, i)) return n;
    if (const std::size_t n = white_space_through_line_end(text, i)) return n;
    if (const std::size_t n = white_space_not_before_text(text, i)) return n;
    return white_space(text, i);
}

}  // namespace

SplitError split(const PreTokenizer& pretokenizer, std::string_view text, std::vector<Piece>& out) {
    if (text.size() > std::numeric_limits<std::uint32_t>::max()) return SplitError::TooLong;
    std::vector<Char> chars;
    chars.reserve(text.size());
    for (std::size_t at = 0; at < text.size();) {
        Utf8Char c{};
        if (!decode_utf8(text, at, c)) return SplitError::InvalidUtf8;
        chars.push_back({c.code_point, char_class(c.code_point), static_cast<std::uint32_t>(at)});
        at += c.length;
    }

    std::vector<Piece> pieces;
    const auto offset = [&](std::size_t i) {
        return i < chars.size() ? chars[i].offset : static_cast<std::uint32_t>(text.size());
    };
    for (std::size_t i = 0; i < chars.size();) {
        // Every character is a letter, a number, white space or a symbol, and
        // some alternative matches each, so a piece is never empty.
        const std::size_t length = piece_at(chars, i, pretokenizer.digits);
        pieces.push_back({offset(i), offset(i + length) - offset(i)});
        i += length;
    }
    out = std::move(pieces);
    return SplitError::Ok;
}

}  // namespace bllm::tokenizer

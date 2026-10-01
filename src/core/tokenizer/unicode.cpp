#include "core/tokenizer/unicode.h"

#include <algorithm>

#include "core/tokenizer/unicode_tables.h"

namespace bllm::tokenizer {

CharClass char_class(char32_t code_point) noexcept {
    const auto table = unicode_tables::classes();
    // The first range that ends at or after the code point; it holds the code
    // point if it also starts at or before it.
    const auto range = std::lower_bound(
        table.begin(), table.end(), code_point,
        [](const unicode_tables::ClassRange& r, char32_t c) { return r.last < c; });
    return range != table.end() && range->first <= code_point ? range->char_class : CharClass::Other;
}

// The well-formed byte sequences of the Unicode Standard, Table 3-7. The lead
// byte fixes the length and narrows the second byte's range; that narrowing
// is what rejects overlong encodings, surrogates and values past U+10FFFF.
bool decode_utf8(std::string_view text, std::size_t at, Utf8Char& out) noexcept {
    const auto byte = [&](std::size_t i) { return static_cast<unsigned char>(text[i]); };
    const unsigned char lead = byte(at);
    if (lead < 0x80) {
        out = {lead, 1};
        return true;
    }
    std::uint8_t length = 0;
    char32_t value = 0;
    unsigned char low = 0x80;    // the range the second byte must lie in
    unsigned char high = 0xBF;
    if (lead >= 0xC2 && lead <= 0xDF) {
        length = 2;
        value = lead & 0x1Fu;
    } else if (lead >= 0xE0 && lead <= 0xEF) {
        length = 3;
        value = lead & 0x0Fu;
        if (lead == 0xE0) low = 0xA0;    // below is overlong
        if (lead == 0xED) high = 0x9F;   // above is a surrogate
    } else if (lead >= 0xF0 && lead <= 0xF4) {
        length = 4;
        value = lead & 0x07u;
        if (lead == 0xF0) low = 0x90;    // below is overlong
        if (lead == 0xF4) high = 0x8F;   // above is past U+10FFFF
    } else {
        return false;   // a continuation byte, an overlong lead, or past U+10FFFF
    }
    if (text.size() - at < length) return false;
    for (std::uint8_t i = 1; i < length; ++i) {
        const unsigned char next = byte(at + i);
        if (next < (i == 1 ? low : 0x80) || next > (i == 1 ? high : 0xBF)) return false;
        value = (value << 6) | (next & 0x3Fu);
    }
    out = {value, length};
    return true;
}

void append_utf8(char32_t code_point, std::string& out) {
    const auto put = [&](char32_t bits) { out.push_back(static_cast<char>(bits)); };
    if (code_point < 0x80) {
        put(code_point);
    } else if (code_point < 0x800) {
        put(0xC0 | (code_point >> 6));
        put(0x80 | (code_point & 0x3F));
    } else if (code_point < 0x10000) {
        put(0xE0 | (code_point >> 12));
        put(0x80 | ((code_point >> 6) & 0x3F));
        put(0x80 | (code_point & 0x3F));
    } else {
        put(0xF0 | (code_point >> 18));
        put(0x80 | ((code_point >> 12) & 0x3F));
        put(0x80 | ((code_point >> 6) & 0x3F));
        put(0x80 | (code_point & 0x3F));
    }
}

}  // namespace bllm::tokenizer

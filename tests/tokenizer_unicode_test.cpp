#include <doctest/doctest.h>

#include <span>
#include <string>
#include <string_view>

#include "core/tokenizer/unicode.h"
#include "core/tokenizer/unicode_tables.h"

using namespace bllm::tokenizer;

namespace {

// Each table is searched by bisection, so it must be sorted with no overlap.
template <typename Range>
void check_disjoint_ascending(std::span<const Range> table) {
    for (std::size_t i = 0; i < table.size(); ++i) {
        CHECK(table[i].first <= table[i].last);
        if (i > 0) CHECK(table[i - 1].last < table[i].first);
    }
}

bool decodes(std::string_view bytes, char32_t expected, std::uint8_t length) {
    Utf8Char c{};
    return decode_utf8(bytes, 0, c) && c.code_point == expected && c.length == length;
}

bool rejects(std::string_view bytes) {
    Utf8Char c{};
    return !decode_utf8(bytes, 0, c);
}

}  // namespace

TEST_CASE("the tables are one Unicode version's, sorted and disjoint") {
    CHECK(unicode_tables::version() == "16.0.0");
    check_disjoint_ascending(unicode_tables::classes());
    check_disjoint_ascending(unicode_tables::combining());
    const auto decompositions = unicode_tables::decompositions();
    for (std::size_t i = 1; i < decompositions.size(); ++i) {
        CHECK(decompositions[i - 1].code_point < decompositions[i].code_point);
    }
    for (const auto& d : decompositions) {
        CHECK(d.length > 0);
        CHECK(std::size_t{d.start} + d.length <= unicode_tables::decomposition_pool().size());
    }
    const auto compositions = unicode_tables::compositions();
    for (std::size_t i = 1; i < compositions.size(); ++i) {
        const auto& a = compositions[i - 1];
        const auto& b = compositions[i];
        CHECK((a.first < b.first || (a.first == b.first && a.second < b.second)));
    }
}

TEST_CASE("every ASCII character is in the class the split patterns expect") {
    for (char32_t c = 0; c < 0x80; ++c) {
        CAPTURE(static_cast<std::uint32_t>(c));
        const bool letter = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
        const bool digit = c >= '0' && c <= '9';
        const bool space = c == ' ' || (c >= 0x09 && c <= 0x0D);
        const CharClass expected = letter ? CharClass::Letter
                                 : digit  ? CharClass::Number
                                 : space  ? CharClass::Whitespace
                                          : CharClass::Other;
        CHECK(char_class(c) == expected);
    }
}

TEST_CASE("classes follow the Unicode properties, not a language's notion of a space") {
    CHECK(char_class(U'é') == CharClass::Letter);
    CHECK(char_class(U'中') == CharClass::Letter);        // inside a First/Last range
    CHECK(char_class(0x20000) == CharClass::Letter);      // CJK Extension B
    CHECK(char_class(0xAC00) == CharClass::Letter);       // a Hangul syllable
    CHECK(char_class(0x0660) == CharClass::Number);       // Arabic-Indic zero, Nd
    CHECK(char_class(0x2167) == CharClass::Number);       // Roman numeral eight, Nl
    CHECK(char_class(0x00BD) == CharClass::Number);       // one half, No
    CHECK(char_class(0x00A0) == CharClass::Whitespace);   // no-break space
    CHECK(char_class(0x0085) == CharClass::Whitespace);   // next line
    CHECK(char_class(0x3000) == CharClass::Whitespace);   // ideographic space
    // Python's str.isspace() calls these spaces; White_Space does not.
    for (char32_t c = 0x1C; c <= 0x1F; ++c) CHECK(char_class(c) == CharClass::Other);
    CHECK(char_class(0x200B) == CharClass::Other);        // zero-width space is a format character
    CHECK(char_class(0x0301) == CharClass::Other);        // a combining accent is a mark
    CHECK(char_class(0x1F600) == CharClass::Other);       // an emoji is a symbol
    CHECK(char_class(0x0378) == CharClass::Other);        // unassigned
    CHECK(char_class(0x10FFFF) == CharClass::Other);
}

TEST_CASE("well-formed UTF-8 decodes, one to four bytes") {
    CHECK(decodes("A", U'A', 1));
    CHECK(decodes("\xC3\xA9", U'é', 2));
    CHECK(decodes("\xE4\xB8\xAD", U'中', 3));
    CHECK(decodes("\xF0\x9F\x98\x80", 0x1F600, 4));
    CHECK(decodes("\xF4\x8F\xBF\xBF", 0x10FFFF, 4));

    // Decoding starts where it is told to.
    Utf8Char c{};
    REQUIRE(decode_utf8("a\xC3\xA9", 1, c));
    CHECK(c.code_point == U'é');
}

TEST_CASE("ill-formed UTF-8 is refused, whatever makes it so") {
    CHECK(rejects("\x80"));               // a continuation byte with no lead
    CHECK(rejects("\xC0\x80"));           // overlong two-byte NUL
    CHECK(rejects("\xC1\xBF"));           // overlong two-byte
    CHECK(rejects("\xE0\x80\x80"));       // overlong three-byte
    CHECK(rejects("\xF0\x80\x80\x80"));   // overlong four-byte
    CHECK(rejects("\xED\xA0\x80"));       // a surrogate
    CHECK(rejects("\xF4\x90\x80\x80"));   // past U+10FFFF
    CHECK(rejects("\xF5\x80\x80\x80"));   // a lead no sequence starts with
    CHECK(rejects("\xE4\xB8"));           // truncated
    CHECK(rejects("\xE4\x41\x80"));       // a lead followed by a non-continuation
}

TEST_CASE("every scalar value encodes and decodes back to itself") {
    for (char32_t c = 0; c <= 0x10FFFF; ++c) {
        if (c >= 0xD800 && c <= 0xDFFF) continue;   // surrogates are not scalar values
        std::string bytes;
        append_utf8(c, bytes);
        Utf8Char decoded{};
        if (!decode_utf8(bytes, 0, decoded) || decoded.code_point != c || decoded.length != bytes.size()) {
            FAIL("U+" << std::hex << static_cast<std::uint32_t>(c) << " does not round-trip");
        }
    }
}

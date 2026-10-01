#include <doctest/doctest.h>

#include <string>
#include <string_view>
#include <vector>

#include "core/tokenizer/tokenizer.h"
#include "core/tokenizer/unicode.h"

using namespace bllm::tokenizer;

namespace {

#define FFFD "\xEF\xBF\xBD"

struct StreamCase {
    std::string_view name;
    std::string_view bytes;
    std::string_view text;   // as Python's bytes.decode("utf-8", "replace") gives it
    bool whole;              // what finish returns: no character left unfinished
};

const StreamCase kCases[] = {
    // The worked example of the Unicode Standard, section 3.9.
    {"the standard's example", "a\xF1\x80\x80\xE1\x80\xC2" "b\x80" "c\x80\xBF" "d",
     "a" FFFD FFFD FFFD "b" FFFD "c" FFFD FFFD "d", true},
    {"one character of each length", "$\xC2\xA2\xE2\x82\xAC\xF0\x90\x8D\x88", "$\xC2\xA2\xE2\x82\xAC\xF0\x90\x8D\x88",
     true},
    {"a two-byte overlong", "\xC0\x80", FFFD FFFD, true},
    {"a three-byte overlong", "\xE0\x80\x80", FFFD FFFD FFFD, true},
    {"a four-byte overlong", "\xF0\x80\x80\x80", FFFD FFFD FFFD FFFD, true},
    {"a surrogate", "\xED\xA0\x80", FFFD FFFD FFFD, true},
    {"past U+10FFFF", "\xF4\x90\x80\x80", FFFD FFFD FFFD FFFD, true},
    {"bytes that begin nothing", "\xF5\xFF\x80", FFFD FFFD FFFD, true},
    {"a character broken off by a letter", "\xE2\x82" "A", FFFD "A", true},
    {"a character broken off by another", "\xF0\x9F\xE2\x82\xAC", FFFD "\xE2\x82\xAC", true},
    {"a character the bytes end inside", "x\xF0\x9F\x98", "x" FFFD, false},
};

#undef FFFD

// Every string a push emits is whole characters: well-formed UTF-8 throughout.
bool well_formed(std::string_view s) {
    for (std::size_t at = 0; at < s.size();) {
        Utf8Char c{};
        if (!decode_utf8(s, at, c)) return false;
        at += c.length;
    }
    return true;
}

// The stream's text for `chunks` pushed in turn, then finished.
std::string through(const std::vector<std::string_view>& chunks, bool& whole) {
    Utf8Stream stream;
    std::string text;
    for (const std::string_view chunk : chunks) {
        std::string emitted;
        stream.push(chunk, emitted);
        CHECK(well_formed(emitted));
        text += emitted;
    }
    std::string last;
    whole = stream.finish(last);
    CHECK(well_formed(last));
    return text + last;
}

}  // namespace

TEST_CASE("the stream gives the reference text however the bytes are split") {
    for (const StreamCase& c : kCases) {
        CAPTURE(c.name);
        bool whole = false;
        CHECK(through({c.bytes}, whole) == c.text);
        CHECK(whole == c.whole);
        for (std::size_t cut = 0; cut <= c.bytes.size(); ++cut) {
            CAPTURE(cut);
            CHECK(through({c.bytes.substr(0, cut), c.bytes.substr(cut)}, whole) == c.text);
            CHECK(whole == c.whole);
        }
        std::vector<std::string_view> bytes;
        for (std::size_t i = 0; i < c.bytes.size(); ++i) bytes.push_back(c.bytes.substr(i, 1));
        CHECK(through(bytes, whole) == c.text);
        CHECK(whole == c.whole);
    }
}

TEST_CASE("a character split between pushes is held until it is complete") {
    Utf8Stream stream;
    std::string out;
    stream.push("ab\xE2\x82", out);
    CHECK(out == "ab");
    stream.push("\xAC" "c", out);
    CHECK(out == "ab\xE2\x82\xAC" "c");
}

TEST_CASE("finishing inside a character replaces what was held, and the stream starts over") {
    Utf8Stream stream;
    std::string out;
    stream.push("\xE2\x82", out);
    CHECK(out.empty());
    CHECK_FALSE(stream.finish(out));
    CHECK(out == "\xEF\xBF\xBD");
    stream.push("A", out);
    CHECK(stream.finish(out));
    CHECK(out == "\xEF\xBF\xBD" "A");
}

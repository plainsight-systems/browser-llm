#include <doctest/doctest.h>

#include <cstdint>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/reader.h"
#include "core/tokenizer/bpe/byte_map.h"
#include "core/tokenizer/special.h"
#include "core/tokenizer/unicode.h"
#include "core/tokenizer/vocabulary.h"
#include "support/metadata_file.h"
#include "support/model_headers.h"

using namespace bllm;
using namespace bllm::tokenizer;
using bllm::testing::MetadataFile;

namespace {

constexpr std::int32_t kNormal = 1, kControl = 3, kUserDefined = 4;

tokenizer::LoadResult load(const MetadataFile& file, Vocabulary& out) {
    const auto bytes = file.bytes();
    gguf::MemoryByteSource source{bytes};
    gguf::TensorIndex index;
    REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);
    return load_vocabulary(source, index, out);
}

Vocabulary small_vocabulary() {
    Vocabulary v;
    const auto r = load(MetadataFile{}
                            .strings("tokenizer.ggml.tokens", {"a", "b", "ab", "<s>", "<|x|>", "  ", "   "})
                            .int32s("tokenizer.ggml.token_type",
                                    {kNormal, kNormal, kNormal, kControl, kControl, kUserDefined, kUserDefined}),
                        v);
    REQUIRE(r.ok());
    return v;
}

Vocabulary real_vocabulary(std::string_view model) {
    const auto header = bllm::testing::read_model_header(model);
    gguf::MemoryByteSource source{std::as_bytes(std::span{header.bytes}), header.file_size};
    Vocabulary v;
    const auto r = load_vocabulary(source, header.index, v);
    REQUIRE_MESSAGE(r.ok(), r.subject);
    return v;
}

TokenId id(std::uint32_t n) { return static_cast<TokenId>(n); }

// The segments of `text`: ordinary runs as their text, special tokens as
// "#<id>".
std::vector<std::string> segments(const SpecialTokens& special, std::string_view text) {
    std::vector<Segment> out;
    special.segment(text, out);
    std::vector<std::string> shown;
    for (const Segment& s : out) {
        shown.push_back(s.special ? "#" + std::to_string(static_cast<std::uint32_t>(*s.special))
                                  : std::string(text.substr(s.offset, s.length)));
    }
    return shown;
}

}  // namespace

TEST_CASE("a vocabulary gives each token's text and type, and each text's token") {
    const Vocabulary v = small_vocabulary();
    REQUIRE(v.size() == 7);
    CHECK(v.text(id(2)) == "ab");
    CHECK(v.type(id(3)) == TokenType::Control);
    CHECK(v.find("ab") == id(2));
    CHECK(v.find("   ") == id(6));
    CHECK_FALSE(v.find("abc").has_value());
    CHECK_FALSE(v.find("").has_value());
}

TEST_CASE("among many tokens, each text finds its own, and only a token's text finds one") {
    // 20,000 tokens, so slots crowd and lookups probe past one another.
    std::vector<std::string> texts;
    for (int i = 0; i < 20'000; ++i) texts.push_back("t" + std::to_string(i));
    texts.push_back("");   // a token may be empty
    const std::vector<std::int32_t> types(texts.size(), 1);
    Vocabulary v;
    REQUIRE(load(MetadataFile{}
                     .strings("tokenizer.ggml.tokens", std::span<const std::string>{texts})
                     .int32s("tokenizer.ggml.token_type", std::span<const std::int32_t>{types}),
                 v)
                .ok());
    for (std::size_t i = 0; i < texts.size(); ++i) {
        if (v.find(texts[i]) != id(static_cast<std::uint32_t>(i))) FAIL("token " << i << " does not find itself");
    }
    for (int i = 0; i < 20'000; ++i) {
        const std::string n = std::to_string(i);
        // A token's text with a byte more, one less, or one changed.
        for (const std::string& other : {"t" + n + "x", "u" + n, "t" + n.substr(0, n.size() - 1) + "~"}) {
            if (v.find(other).has_value()) FAIL(other << " finds a token");
        }
    }
    CHECK_FALSE(v.find("t").has_value());
    CHECK_FALSE(v.find("t20000").has_value());
    // An empty vocabulary finds nothing.
    CHECK_FALSE(Vocabulary{}.find("t0").has_value());
}

TEST_CASE("a vocabulary that cannot be read as one is refused, by name") {
    Vocabulary v;
    auto r = load(MetadataFile{}.strings("tokenizer.ggml.tokens", {"a"}), v);
    CHECK(r.error == LoadError::MissingKey);
    CHECK(r.subject == "tokenizer.ggml.token_type");

    r = load(MetadataFile{}.int32s("tokenizer.ggml.tokens", {1}).int32s("tokenizer.ggml.token_type", {1}), v);
    CHECK(r.error == LoadError::Unreadable);   // tokens that are not strings

    r = load(MetadataFile{}.strings("tokenizer.ggml.tokens", {"a", "b"}).int32s("tokenizer.ggml.token_type", {1}), v);
    CHECK(r.error == LoadError::CountMismatch);

    r = load(MetadataFile{}.strings("tokenizer.ggml.tokens", {"a", "b"}).int32s("tokenizer.ggml.token_type", {1, 7}), v);
    CHECK(r.error == LoadError::UnknownTokenType);
    CHECK(r.subject == "token 1");

    // The references disagree on which repeat would win, so none is guessed.
    r = load(MetadataFile{}.strings("tokenizer.ggml.tokens", {"a", "b", "a"}).int32s("tokenizer.ggml.token_type", {1, 1, 1}), v);
    CHECK(r.error == LoadError::DuplicateToken);
    CHECK(r.subject == "a");
    CHECK(v.size() == 0);   // untouched
}

TEST_CASE("special tokens are found in raw text, leftmost and longest") {
    const SpecialTokens special{small_vocabulary()};
    CHECK(special.size() == 4);   // control and user-defined; normal tokens are never special
    using S = std::vector<std::string>;
    CHECK(segments(special, "") == S{});
    CHECK(segments(special, "ab") == S{"ab"});
    CHECK(segments(special, "a<s>b") == S{"a", "#3", "b"});
    CHECK(segments(special, "<s><|x|>") == S{"#3", "#4"});
    // "  " is a prefix of "   ": the longer wins, then the search resumes after it.
    CHECK(segments(special, "a   b") == S{"a", "#6", "b"});
    CHECK(segments(special, "a     b") == S{"a", "#6", "#5", "b"});
    // Text that only starts like a special token is ordinary.
    CHECK(segments(special, "<s <|x") == S{"<s <|x"});
}

TEST_CASE("every byte has its own character, and each character gives its byte back") {
    std::set<char32_t> seen;
    for (std::uint32_t byte = 0; byte < 256; ++byte) {
        const char32_t c = bpe::kByteChars[byte];
        CHECK(seen.insert(c).second);
        CHECK(bpe::byte_of(c) == byte);
    }
    CHECK_FALSE(bpe::byte_of(U' ').has_value());     // a space is spelled "Ġ"
    CHECK_FALSE(bpe::byte_of(0x144).has_value());    // past the 68 moved bytes
    // No other character gives a byte: none up to well past the map's end, and
    // none at the far end of the code points.
    for (char32_t c = 0; c < 0x300; ++c) {
        CAPTURE(static_cast<std::uint32_t>(c));
        CHECK(bpe::byte_of(c).has_value() == seen.contains(c));
    }
    CHECK_FALSE(bpe::byte_of(0x10FFFF).has_value());
    CHECK_FALSE(bpe::byte_of(0xFFFFFFFF).has_value());
}

TEST_CASE("each listed model's vocabulary loads, and every text finds its own token") {
    struct Expected {
        std::string_view model;
        std::size_t size;
    };
    for (const Expected e : {Expected{"qwen3-0.6b-q4_0", 151'936}, Expected{"llama-3.2-1b-instruct-q4_0", 128'256},
                             Expected{"gemma-3-1b-it-q4_0", 262'144}}) {
        CAPTURE(e.model);
        const Vocabulary v = real_vocabulary(e.model);
        CHECK(v.size() == e.size);
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (v.find(v.text(id(static_cast<std::uint32_t>(i)))) != id(static_cast<std::uint32_t>(i))) {
                FAIL("token " << i << " does not find itself");
            }
        }
    }
}

TEST_CASE("a byte-level vocabulary has a token for every byte's character") {
    for (const std::string_view model : {"qwen3-0.6b-q4_0", "llama-3.2-1b-instruct-q4_0"}) {
        CAPTURE(model);
        const Vocabulary v = real_vocabulary(model);
        for (std::uint32_t byte = 0; byte < 256; ++byte) {
            std::string text;
            append_utf8(bpe::kByteChars[byte], text);
            CHECK(v.find(text).has_value());
        }
    }
}

TEST_CASE("the listed models' chat-template tokens are special, each one token") {
    const SpecialTokens qwen{real_vocabulary("qwen3-0.6b-q4_0")};
    CHECK(segments(qwen, "<|im_start|>user\nHi<|im_end|>") ==
          std::vector<std::string>{"#151644", "user\nHi", "#151645"});
    const SpecialTokens llama{real_vocabulary("llama-3.2-1b-instruct-q4_0")};
    CHECK(segments(llama, "<|begin_of_text|><|start_header_id|>user<|end_header_id|>") ==
          std::vector<std::string>{"#128000", "#128006", "user", "#128007"});
}

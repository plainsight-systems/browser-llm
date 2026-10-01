#include <doctest/doctest.h>

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/tokenizer/pretokenize.h"

using namespace bllm::tokenizer;

namespace {

struct SplitCase {
    std::string_view name;
    std::string_view text;
    std::vector<std::string_view> pieces;
};

// The pieces Hugging Face tokenizers splits each fixture text into, with the
// pattern from each model's tokenizer.json (tools/make_tokenizer_fixtures.py).
const SplitCase kQwen2Cases[] = {
#include "fixtures/tokenizer/split-qwen2.inc"
};
const SplitCase kLlamaBpeCases[] = {
#include "fixtures/tokenizer/split-llama-bpe.inc"
};

std::vector<std::string_view> pieces(const PreTokenizer& pretokenizer, std::string_view text) {
    std::vector<Piece> split_pieces;
    REQUIRE(split(pretokenizer, text, split_pieces) == SplitError::Ok);
    std::vector<std::string_view> out;
    for (const Piece& p : split_pieces) out.push_back(text.substr(p.offset, p.length));
    return out;
}

void check_cases(const PreTokenizer& pretokenizer, std::span<const SplitCase> cases) {
    for (const SplitCase& c : cases) {
        CAPTURE(c.name);
        CHECK(pieces(pretokenizer, c.text) == c.pieces);
    }
}

}  // namespace

TEST_CASE("qwen2 splits every fixture text as Qwen's tokenizer.json does") {
    check_cases(kQwen2, kQwen2Cases);
}

TEST_CASE("llama-bpe splits every fixture text as Llama's tokenizer.json does") {
    check_cases(kLlamaBpe, kLlamaBpeCases);
}

TEST_CASE("the two patterns differ only in how many digits a piece holds") {
    CHECK(pieces(kQwen2, "2026") == std::vector<std::string_view>{"2", "0", "2", "6"});
    CHECK(pieces(kLlamaBpe, "2026") == std::vector<std::string_view>{"202", "6"});
    CHECK(kQwen2.normalization == Normalization::Nfc);
    CHECK(kLlamaBpe.normalization == Normalization::None);
}

// Edges of the pattern, each checked against Hugging Face tokenizers' split.
TEST_CASE("each alternative takes what the reference split takes") {
    using P = std::vector<std::string_view>;
    // A contraction ending needs no word boundary after it.
    CHECK(pieces(kQwen2, "'sun") == P{"'s", "un"});
    // A curly apostrophe is not a contraction; it leads the letters instead.
    CHECK(pieces(kQwen2, "it’s") == P{"it", "’s"});
    // White space runs to its last line end, whatever lies between.
    CHECK(pieces(kQwen2, "x  \n  \n y") == P{"x", "  \n  \n", " y"});
    // Any white space, not only a space, gives up its last character to the
    // letters after it.
    CHECK(pieces(kQwen2, "a  b") == P{"a", " ", " b"});
    // White space that ends the text is kept whole.
    CHECK(pieces(kQwen2, "  ") == P{"  "});
    // A combining mark is not a letter: it leads the letters after it, and
    // stands alone after the letter it modifies (until NFC composes them).
    CHECK(pieces(kQwen2, "́abc") == P{"́abc"});
    CHECK(pieces(kQwen2, "é") == P{"e", "́"});
}

TEST_CASE("text that is not UTF-8 is refused, and the output is untouched") {
    std::vector<Piece> out{{7, 7}};
    CHECK(split(kQwen2, "ok\xC0\x80", out) == SplitError::InvalidUtf8);
    CHECK(out.size() == 1);
    CHECK(out[0].offset == 7);
}

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string_view>
#include <vector>

#include "core/gguf/reader.h"
#include "core/tokenizer/bpe/merge_table.h"
#include "core/tokenizer/bpe/sentencepiece_merges.h"
#include "core/tokenizer/load.h"
#include "core/tokenizer/vocabulary.h"
#include "support/metadata_file.h"
#include "support/model_headers.h"

using namespace bllm;
using namespace bllm::tokenizer;
using bllm::testing::MetadataFile;
using bpe::MergeTable;

namespace {

constexpr std::int32_t kNormal = 1;
constexpr std::int32_t kControl = 3;

Vocabulary vocabulary_of(std::initializer_list<std::string_view> tokens, std::initializer_list<std::int32_t> types) {
    const auto bytes = MetadataFile{}
                           .strings("tokenizer.ggml.tokens", tokens)
                           .int32s("tokenizer.ggml.token_type", types)
                           .bytes();
    gguf::MemoryByteSource source{bytes};
    gguf::TensorIndex index;
    REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);
    Vocabulary vocabulary;
    REQUIRE(load_vocabulary(source, index, vocabulary).ok());
    return vocabulary;
}

MergeTable implied(const Vocabulary& vocabulary, std::vector<float> scores) {
    MergeTable table;
    const auto r = bpe::implied_merges(vocabulary, scores, table);
    REQUIRE_MESSAGE(r.ok(), r.subject);
    return table;
}

// The rule for two tokens named by their text, if there is one.
std::optional<bpe::Merge> rule(const Vocabulary& v, MergeTable& table, std::string_view left,
                               std::string_view right) {
    return table.find(*v.find(left), *v.find(right));
}

}  // namespace

TEST_CASE("every split of a token into two tokens is a rule, ranked by the score of the token it makes") {
    const auto v = vocabulary_of({"a", "b", "c", "ab", "bc", "abc"}, {1, 1, 1, 1, 1, 1});
    auto table = implied(v, {0, 0, 0, -2, -1, -3});
    CHECK(table.size() == 4);
    CHECK(rule(v, table, "a", "b")->result == *v.find("ab"));
    CHECK(rule(v, table, "b", "c")->rank < rule(v, table, "a", "b")->rank);   // bc scores higher
    // abc is made two ways, at one rank.
    CHECK(rule(v, table, "a", "bc")->result == *v.find("abc"));
    CHECK(rule(v, table, "ab", "c")->result == *v.find("abc"));
    CHECK(rule(v, table, "a", "bc")->rank == rule(v, table, "ab", "c")->rank);
    CHECK(rule(v, table, "a", "b")->rank < rule(v, table, "a", "bc")->rank);
    CHECK_FALSE(rule(v, table, "b", "a").has_value());
}

TEST_CASE("tokens of equal score make rules of one rank") {
    const auto v = vocabulary_of({"a", "b", "c", "ab", "bc"}, {1, 1, 1, 1, 1});
    auto table = implied(v, {0, 0, 0, -1, -1});
    CHECK(rule(v, table, "a", "b")->rank == rule(v, table, "b", "c")->rank);
}

TEST_CASE("a token splits only between characters") {
    // "é" is two bytes. Its first byte alone and "x" after its second are
    // tokens here, but a split inside a character is no split of the text.
    const auto v = vocabulary_of({"\xC3\xA9", "x", "\xC3\xA9x", "\xC3", "\xA9x"}, {1, 1, 1, 1, 1});
    auto table = implied(v, {0, 0, -1, 0, -2});
    CHECK(table.size() == 1);
    CHECK(rule(v, table, "\xC3\xA9", "x")->result == *v.find("\xC3\xA9x"));
}

TEST_CASE("only normal tokens take part in a rule") {
    // A control token made of two normal ones is made by no rule.
    auto v = vocabulary_of({"a", "b", "ab"}, {kNormal, kNormal, kControl});
    CHECK(implied(v, {0, 0, -1}).size() == 0);
    // Nor is a normal token made of a control one and a normal one.
    v = vocabulary_of({"a", "b", "ab"}, {kControl, kNormal, kNormal});
    CHECK(implied(v, {0, 0, -1}).size() == 0);
}

TEST_CASE("a score that cannot be ordered is refused, and scores must match the tokens") {
    const auto v = vocabulary_of({"a", "b", "ab", "<x>"}, {kNormal, kNormal, kNormal, kControl});
    MergeTable table;
    auto r = bpe::implied_merges(v, std::vector<float>{0, NAN, -1, 0}, table);
    CHECK(r.error == LoadError::InvalidScore);
    CHECK(r.subject == "b");
    CHECK(table.size() == 0);   // untouched
    // A control token is never ordered, so its score does not matter.
    CHECK(bpe::implied_merges(v, std::vector<float>{0, 0, -1, NAN}, table).ok());
    CHECK(bpe::implied_merges(v, std::vector<float>{0, 0, -1}, table).error == LoadError::CountMismatch);
}

TEST_CASE("Gemma 3's vocabulary implies Hugging Face's merges for it") {
    const auto header = bllm::testing::read_model_header("gemma-3-1b-it-q4_0");
    gguf::MemoryByteSource source{std::as_bytes(std::span{header.bytes}), header.file_size};
    Vocabulary v;
    REQUIRE(load_vocabulary(source, header.index, v).ok());
    std::vector<float> scores;
    REQUIRE(read_float32_array(source, header.index, "tokenizer.ggml.scores", scores).ok());
    auto table = implied(v, scores);
    // tokenizer.json lists 514,906 merges for Gemma 3. 513,511 make normal
    // tokens, and they are exactly these; the rest make runs of tabs, newlines
    // and spaces, which are user-defined tokens, matched before merging.
    CHECK(table.size() == 513'511);
    // " the" is made three ways, all at one rank.
    const auto the = rule(v, table, "\xE2\x96\x81th", "e");
    REQUIRE(the.has_value());
    CHECK(the->result == *v.find("\xE2\x96\x81the"));
    CHECK(rule(v, table, "\xE2\x96\x81t", "he")->rank == the->rank);
    CHECK(rule(v, table, "\xE2\x96\x81", "the")->rank == the->rank);
}

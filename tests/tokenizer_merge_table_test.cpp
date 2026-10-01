#include <doctest/doctest.h>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "core/gguf/reader.h"
#include "core/tokenizer/bpe/merge_table.h"
#include "core/tokenizer/vocabulary.h"
#include "support/metadata_file.h"
#include "support/model_headers.h"

using namespace bllm;
using namespace bllm::tokenizer;
using bllm::testing::MetadataFile;
using bpe::MergeRule;
using bpe::MergeTable;

namespace {

TokenId id(std::uint32_t n) { return static_cast<TokenId>(n); }

// A vocabulary of a, b, c, ab, abc, read alongside `merges`.
LoadResult load_with_merges(std::initializer_list<std::string_view> merges, MergeTable& out) {
    const auto bytes = MetadataFile{}
                           .strings("tokenizer.ggml.tokens", {"a", "b", "c", "ab", "abc"})
                           .int32s("tokenizer.ggml.token_type", {1, 1, 1, 1, 1})
                           .strings("tokenizer.ggml.merges", merges)
                           .bytes();
    gguf::MemoryByteSource source{bytes};
    gguf::TensorIndex index;
    REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);
    Vocabulary vocabulary;
    REQUIRE(load_vocabulary(source, index, vocabulary).ok());
    return bpe::load_merges(source, index, vocabulary, out);
}

}  // namespace

TEST_CASE("a merge table finds each rule's rank and result by its pair") {
    const MergeRule rules[] = {{id(1), id(2), id(9)}, {id(0), id(1), id(8)}};
    const auto table = MergeTable::from_rules(rules);
    REQUIRE(table.has_value());
    CHECK(table->size() == 2);
    const auto bc = table->find(id(1), id(2));
    REQUIRE(bc.has_value());
    CHECK(bc->rank == 0);
    CHECK(bc->result == id(9));
    CHECK(table->find(id(0), id(1))->rank == 1);
    CHECK_FALSE(table->find(id(2), id(1)).has_value());   // order matters
}

TEST_CASE("rules that merge the same pair twice make no table") {
    const MergeRule rules[] = {{id(0), id(1), id(8)}, {id(0), id(1), id(9)}};
    CHECK_FALSE(MergeTable::from_rules(rules).has_value());
}

TEST_CASE("merges load from the file, each pair's text joined to its result") {
    MergeTable table;
    REQUIRE(load_with_merges({"a b", "ab c"}, table).ok());
    CHECK(table.size() == 2);
    CHECK(table.find(id(0), id(1))->result == id(3));   // a b -> ab
    CHECK(table.find(id(3), id(2))->result == id(4));   // ab c -> abc
    CHECK(table.find(id(3), id(2))->rank == 1);
}

TEST_CASE("a merge that is not two known tokens is refused, by position") {
    MergeTable table;
    for (const std::string_view malformed : {"ab", " a", "a ", "a b c", ""}) {
        CAPTURE(malformed);
        const auto r = load_with_merges({"a b", malformed}, table);
        CHECK(r.error == LoadError::MalformedMerge);
        CHECK(r.subject == "merge 1");
    }
    // A token the vocabulary lacks, and a pair whose join it lacks ("ac").
    CHECK(load_with_merges({"a d"}, table).error == LoadError::UnknownMergeToken);
    CHECK(load_with_merges({"a c"}, table).error == LoadError::UnknownMergeToken);
    CHECK(load_with_merges({"a b", "a b"}, table).error == LoadError::DuplicateMerge);
    CHECK(table.size() == 0);   // untouched
}

TEST_CASE("the byte-level models' merges load in full") {
    struct Expected {
        std::string_view model;
        std::size_t merges;
    };
    for (const Expected e : {Expected{"qwen3-0.6b-q4_0", 151'387}, Expected{"llama-3.2-1b-instruct-q4_0", 280'147}}) {
        CAPTURE(e.model);
        const auto header = bllm::testing::read_model_header(e.model);
        gguf::MemoryByteSource source{std::as_bytes(std::span{header.bytes}), header.file_size};
        Vocabulary vocabulary;
        REQUIRE(load_vocabulary(source, header.index, vocabulary).ok());
        MergeTable table;
        const auto r = bpe::load_merges(source, header.index, vocabulary, table);
        REQUIRE_MESSAGE(r.ok(), r.subject);
        CHECK(table.size() == e.merges);
        // Both lists begin with two spaces merging: "Ġ Ġ" -> "ĠĠ".
        const auto space = vocabulary.find("\xC4\xA0");
        REQUIRE(space.has_value());
        const auto first = table.find(*space, *space);
        REQUIRE(first.has_value());
        CHECK(first->rank == 0);
        CHECK(vocabulary.text(first->result) == "\xC4\xA0\xC4\xA0");
    }
}

#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "core/tokenizer/bpe/piece_cache.h"

using namespace bllm::tokenizer;
using bpe::PieceCache;

namespace {

std::vector<TokenId> tokens(std::initializer_list<std::uint32_t> ids) {
    std::vector<TokenId> out;
    for (const std::uint32_t id : ids) out.push_back(static_cast<TokenId>(id));
    return out;
}

std::vector<TokenId> found(const PieceCache& cache, std::string_view piece) {
    const auto hit = cache.find(piece);
    REQUIRE(hit.has_value());
    return {hit->begin(), hit->end()};
}

}  // namespace

TEST_CASE("a cache returns what was recorded for a piece, and nothing else") {
    PieceCache cache{64};
    CHECK_FALSE(cache.find("hello").has_value());
    cache.insert("hello", tokens({9}));
    cache.insert(" world", tokens({1, 2, 3}));
    CHECK(found(cache, "hello") == tokens({9}));
    CHECK(found(cache, " world") == tokens({1, 2, 3}));
    CHECK_FALSE(cache.find("hell").has_value());     // a prefix is another piece
    CHECK_FALSE(cache.find("hello!").has_value());
    cache.clear();
    CHECK_FALSE(cache.find("hello").has_value());
}

TEST_CASE("pieces or results too long for a slot are not recorded") {
    PieceCache cache{64};
    const std::string sixteen(PieceCache::kMaxBytes + 1, 'x');
    cache.insert(sixteen, tokens({1}));
    CHECK_FALSE(cache.find(sixteen).has_value());
    cache.insert("four", tokens({1, 2, 3, 4}));      // one token more than a slot holds
    CHECK_FALSE(cache.find("four").has_value());
    cache.insert("", tokens({1}));
    CHECK_FALSE(cache.find("").has_value());
}

TEST_CASE("pieces that share a slot replace one another") {
    PieceCache cache{1};
    cache.insert("first", tokens({1}));
    cache.insert("second", tokens({2}));
    CHECK_FALSE(cache.find("first").has_value());
    CHECK(found(cache, "second") == tokens({2}));
}

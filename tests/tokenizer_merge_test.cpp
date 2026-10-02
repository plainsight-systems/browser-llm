#include <doctest/doctest.h>

#include <cstdint>
#include <initializer_list>
#include <vector>

#include "core/tokenizer/bpe/merge.h"
#include "core/tokenizer/bpe/merge_table.h"

using namespace bllm::tokenizer;
using bpe::MergeRule;
using bpe::MergeTable;

namespace {

// Tokens named for what they spell.
enum : std::uint32_t { a, b, c, ab, bc, abc, aa, aaaa, x };

TokenId t(std::uint32_t n) { return static_cast<TokenId>(n); }

MergeTable table(std::initializer_list<MergeRule> rules) {
    auto made = MergeTable::from_rules(std::vector<MergeRule>(rules));
    REQUIRE(made.has_value());
    return *made;
}

MergeRule rule(std::uint32_t left, std::uint32_t right, std::uint32_t result) {
    return {t(left), t(right), t(result)};
}

MergeTable ranked(std::initializer_list<bpe::RankedRule> rules) {
    auto made = MergeTable::from_ranked_rules(std::vector<bpe::RankedRule>(rules));
    REQUIRE(made.has_value());
    return *made;
}

std::vector<TokenId> tokens(std::initializer_list<std::uint32_t> ids) {
    std::vector<TokenId> out;
    for (const std::uint32_t id : ids) out.push_back(t(id));
    return out;
}

// Merges `symbols` both ways: as they are, a short piece, and after 100
// symbols of x, which no rule takes, making a piece long enough for the heap.
// Both must agree; the result is the short one.
std::vector<TokenId> merged(const MergeTable& merges, std::initializer_list<std::uint32_t> symbols) {
    std::vector<TokenId> short_out;
    bpe::merge(tokens(symbols), merges, short_out);

    std::vector<TokenId> padded(100, t(x));
    const auto piece = tokens(symbols);
    padded.insert(padded.end(), piece.begin(), piece.end());
    std::vector<TokenId> long_out;
    bpe::merge(padded, merges, long_out);
    CHECK(std::vector<TokenId>(long_out.begin(), long_out.begin() + 100) == std::vector<TokenId>(100, t(x)));
    CHECK(std::vector<TokenId>(long_out.begin() + 100, long_out.end()) == short_out);
    return short_out;
}

}  // namespace

TEST_CASE("nothing to merge leaves the symbols as they are") {
    const MergeTable merges = table({rule(a, b, ab)});
    CHECK(merged(merges, {}).empty());
    CHECK(merged(merges, {a}) == tokens({a}));
    CHECK(merged(merges, {b, a}) == tokens({b, a}));   // the rule is for a then b
    CHECK(merged(MergeTable{}, {a, b, c}) == tokens({a, b, c}));
}

TEST_CASE("a pair with a rule becomes the token the rule makes") {
    CHECK(merged(table({rule(a, b, ab)}), {a, b}) == tokens({ab}));
    CHECK(merged(table({rule(a, b, ab)}), {c, a, b, c}) == tokens({c, ab, c}));
}

TEST_CASE("the lowest rank merges first, wherever it is") {
    // b c ranks above a b, so it merges first, though a b is to its left.
    CHECK(merged(table({rule(b, c, bc), rule(a, b, ab)}), {a, b, c}) == tokens({a, bc}));
    CHECK(merged(table({rule(a, b, ab), rule(b, c, bc)}), {a, b, c}) == tokens({ab, c}));
}

TEST_CASE("a merged token merges again when a rule takes it") {
    CHECK(merged(table({rule(a, b, ab), rule(ab, c, abc)}), {a, b, c}) == tokens({abc}));
}

TEST_CASE("each round takes the lowest rank among the pairs that exist now") {
    // b c merges first; then a bc (rank 1) outranks a b (rank 2), which no
    // longer exists anyway. Merging left to right would give ab c instead.
    const MergeTable merges = table({rule(b, c, bc), rule(a, bc, abc), rule(a, b, ab)});
    CHECK(merged(merges, {a, b, c}) == tokens({abc}));
}

TEST_CASE("a repeated pair merges leftmost first, without overlapping") {
    CHECK(merged(table({rule(a, a, aa)}), {a, a, a}) == tokens({aa, a}));
    CHECK(merged(table({rule(a, a, aa)}), {a, a, a, a}) == tokens({aa, aa}));
    CHECK(merged(table({rule(a, a, aa), rule(aa, aa, aaaa)}), {a, a, a, a, a}) == tokens({aaaa, a}));
}

TEST_CASE("two pairs whose rules share a rank merge leftmost first") {
    // a b and b c rank alike: a b is to the left, so it merges, and b c no
    // longer exists. The order the rules are given in does not matter.
    CHECK(merged(ranked({{rule(a, b, ab), 0}, {rule(b, c, bc), 0}}), {a, b, c}) == tokens({ab, c}));
    CHECK(merged(ranked({{rule(b, c, bc), 0}, {rule(a, b, ab), 0}}), {a, b, c}) == tokens({ab, c}));
    // A lower rank still wins, wherever it is.
    CHECK(merged(ranked({{rule(a, b, ab), 1}, {rule(b, c, bc), 0}}), {a, b, c}) == tokens({a, bc}));
}

TEST_CASE("merged tokens are appended after what the output holds") {
    std::vector<TokenId> out = tokens({x});
    bpe::merge(tokens({a, b}), table({rule(a, b, ab)}), out);
    CHECK(out == tokens({x, ab}));
}

TEST_CASE("a long piece merges without quadratic cost") {
    // 200,000 symbols: rescanning every pair after every merge would be tens
    // of billions of steps. Pairs of a merge to aa; pairs of aa to aaaa.
    const MergeTable merges = table({rule(a, a, aa), rule(aa, aa, aaaa)});
    const std::vector<TokenId> symbols(200'000, t(a));
    std::vector<TokenId> out;
    bpe::merge(symbols, merges, out);
    CHECK(out == std::vector<TokenId>(50'000, t(aaaa)));
}

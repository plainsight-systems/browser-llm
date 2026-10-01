#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/tokenizer/tokenizer.h"
#include "core/tokenizer/vocabulary.h"

namespace bllm::tokenizer::bpe {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// A BPE model's merge rules: which two adjacent tokens merge, into which
// token, and how early. A rule's rank is its place in the list; the lower
// rank merges first.

struct MergeRule {
    TokenId left;
    TokenId right;
    TokenId result;
};

struct Merge {
    std::uint32_t rank;
    TokenId result;
};

class MergeTable {
public:
    MergeTable() = default;

    // A table of `rules` in rank order, the first ranked 0. Empty if two
    // rules merge the same pair: the references disagree on which would win,
    // and no vocabulary listed here repeats one.
    [[nodiscard]] static std::optional<MergeTable> from_rules(std::span<const MergeRule> rules);

    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

    // The rule for `left` followed by `right`, if there is one.
    [[nodiscard]] std::optional<Merge> find(TokenId left, TokenId right) const noexcept;

private:
    struct Entry {
        std::uint64_t pair;   // left in the high 32 bits, right in the low
        Merge merge;
    };

    // Optimization (practice): a sorted vector of packed 64-bit pairs searched by
    // bisection, contiguous where std::unordered_map allocates a node per entry
    // (CACHE.3). Hugging Face tokenizers goes further, with a perfect hash.
    std::vector<Entry> entries_;   // ordered by pair
};

// Reads tokenizer.ggml.merges: each "left right", two tokens' text separated
// by one space, merging to the token whose text is both. A byte-level
// vocabulary spells a space as "Ġ", so no token text holds one. Every token
// named must be in `vocabulary`. `out` is left untouched unless the read
// succeeds.
[[nodiscard]] LoadResult load_merges(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                     const Vocabulary& vocabulary, MergeTable& out);

}  // namespace bllm::tokenizer::bpe

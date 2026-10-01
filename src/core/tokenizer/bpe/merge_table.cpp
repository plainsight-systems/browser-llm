#include "core/tokenizer/bpe/merge_table.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>

#include "core/tokenizer/load.h"

namespace bllm::tokenizer::bpe {
namespace {

std::uint64_t pair_key(TokenId left, TokenId right) {
    return (std::uint64_t{static_cast<std::uint32_t>(left)} << 32) | static_cast<std::uint32_t>(right);
}

}  // namespace

std::optional<MergeTable> MergeTable::from_rules(std::span<const MergeRule> rules) {
    MergeTable table;
    table.entries_.reserve(rules.size());
    for (std::size_t rank = 0; rank < rules.size(); ++rank) {
        const MergeRule& r = rules[rank];
        table.entries_.push_back({pair_key(r.left, r.right), {static_cast<std::uint32_t>(rank), r.result}});
    }
    std::sort(table.entries_.begin(), table.entries_.end(),
              [](const Entry& a, const Entry& b) { return a.pair < b.pair; });
    const auto repeat = std::adjacent_find(table.entries_.begin(), table.entries_.end(),
                                           [](const Entry& a, const Entry& b) { return a.pair == b.pair; });
    if (repeat != table.entries_.end()) return std::nullopt;
    return table;
}

std::optional<Merge> MergeTable::find(TokenId left, TokenId right) const noexcept {
    const std::uint64_t key = pair_key(left, right);
    const auto at = std::lower_bound(entries_.begin(), entries_.end(), key,
                                     [](const Entry& e, std::uint64_t k) { return e.pair < k; });
    if (at == entries_.end() || at->pair != key) return std::nullopt;
    return at->merge;
}

LoadResult load_merges(gguf::ByteSource& source, const gguf::TensorIndex& index, const Vocabulary& vocabulary,
                       MergeTable& out) {
    constexpr std::string_view kMerges = "tokenizer.ggml.merges";
    gguf::StringTable merges;
    if (auto r = read_string_array(source, index, kMerges, merges); !r.ok()) return r;

    std::vector<MergeRule> rules;
    rules.reserve(merges.size());
    std::string joined;
    for (std::size_t i = 0; i < merges.size(); ++i) {
        const std::string_view merge = merges[i];
        const std::size_t space = merge.find(' ');
        if (space == std::string_view::npos || space == 0 || space + 1 == merge.size() ||
            merge.find(' ', space + 1) != std::string_view::npos) {
            return {LoadError::MalformedMerge, "merge " + std::to_string(i)};
        }
        const std::string_view left = merge.substr(0, space);
        const std::string_view right = merge.substr(space + 1);
        joined.assign(left).append(right);
        const auto l = vocabulary.find(left);
        const auto r = vocabulary.find(right);
        const auto result = vocabulary.find(joined);
        if (!l || !r || !result) return {LoadError::UnknownMergeToken, "merge " + std::to_string(i)};
        rules.push_back({*l, *r, *result});
    }
    auto table = MergeTable::from_rules(rules);
    if (!table) return {LoadError::DuplicateMerge, std::string(kMerges)};
    out = std::move(*table);
    return {};
}

}  // namespace bllm::tokenizer::bpe

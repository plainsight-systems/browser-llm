#include "core/tokenizer/bpe/merge.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <queue>

namespace bllm::tokenizer::bpe {
namespace {

constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();

// One symbol of the piece, linked to its neighbours by position. A merge
// gives the left symbol the merged token and unlinks the right one, which no
// pair ever reaches again.
struct Symbol {
    TokenId token;
    std::uint32_t prev;
    std::uint32_t next;
};

// An adjacent pair that has a rule, as it stood when it was found. Merges
// around it may since have changed it, so it is checked again before it
// merges.
struct Candidate {
    std::uint32_t rank;
    std::uint32_t left;    // positions; the left one also breaks ties, leftmost first
    std::uint32_t right;
    TokenId left_token;
    TokenId right_token;
    TokenId result;

    // For a min-heap: the lowest rank first, then the leftmost.
    bool operator>(const Candidate& other) const noexcept {
        return rank != other.rank ? rank > other.rank : left > other.left;
    }
};

using Queue = std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>>;

// Queues the pair that starts at `left`, if it has a rule.
void consider(const std::vector<Symbol>& piece, std::uint32_t left, const MergeTable& merges, Queue& queue) {
    const std::uint32_t right = piece[left].next;
    if (right == kNone) return;
    if (const auto rule = merges.find(piece[left].token, piece[right].token)) {
        queue.push({rule->rank, left, right, piece[left].token, piece[right].token, rule->result});
    }
}

// True when the pair still stands as it did when it was queued: both symbols
// linked to each other and holding the same tokens. An unlinked symbol has no
// next, so a pair naming one never stands.
bool still_stands(const std::vector<Symbol>& piece, const Candidate& c) {
    return piece[c.left].next == c.right && piece[c.left].token == c.left_token &&
           piece[c.right].token == c.right_token;
}

}  // namespace

void merge(std::span<const TokenId> symbols, const MergeTable& merges, std::vector<TokenId>& out) {
    const auto count = static_cast<std::uint32_t>(symbols.size());
    if (count == 0) return;

    std::vector<Symbol> piece(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        piece[i] = {symbols[i], i == 0 ? kNone : i - 1, i + 1 == count ? kNone : i + 1};
    }

    Queue queue;
    for (std::uint32_t i = 0; i + 1 < count; ++i) consider(piece, i, merges, queue);

    while (!queue.empty()) {
        const Candidate c = queue.top();
        queue.pop();
        if (!still_stands(piece, c)) continue;

        // The left symbol becomes the merged token and takes over the right
        // one's place in the chain.
        Symbol& left = piece[c.left];
        Symbol& right = piece[c.right];
        left.token = c.result;
        left.next = right.next;
        if (right.next != kNone) piece[right.next].prev = c.left;
        right.prev = right.next = kNone;

        // Only the two pairs that include the new token are new.
        if (left.prev != kNone) consider(piece, left.prev, merges, queue);
        consider(piece, c.left, merges, queue);
    }

    // The first symbol is never the right of a pair, so the chain starts at 0.
    for (std::uint32_t i = 0; i != kNone; i = piece[i].next) out.push_back(piece[i].token);
}

}  // namespace bllm::tokenizer::bpe

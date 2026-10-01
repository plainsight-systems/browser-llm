#pragma once

#include <span>
#include <vector>

#include "core/tokenizer/bpe/merge_table.h"
#include "core/tokenizer/tokenizer.h"

namespace bllm::tokenizer::bpe {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// The merge step of byte-level BPE, for one piece of split text.
//
// `symbols` is the piece as tokens, one for each of its bytes. Merging
// repeatedly finds the adjacent pair whose rule ranks lowest — the rule the
// file lists earliest — and replaces the two with the token the rule makes.
// When the same pair occurs more than once, the leftmost merges first, and a
// pair that shares a token with one already merged no longer exists. Merging
// stops when no adjacent pair has a rule. The tokens left are appended to
// `out`, after whatever it already holds.
//
// One piece can be long: a word of 10,000 letters is one piece. The cost must
// not grow with the square of the piece's length.
//
// Optimization (practice): two strategies by length, as tiktoken and Hugging
// Face tokenizers both choose. A short piece is rescanned, its pair ranks in
// fixed arrays on the stack; a long one goes through a min-heap over a chain
// of symbols, O(n log n), which bounds the cost of a long or hostile piece.
void merge(std::span<const TokenId> symbols, const MergeTable& merges, std::vector<TokenId>& out);

}  // namespace bllm::tokenizer::bpe

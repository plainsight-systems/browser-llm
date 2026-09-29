#pragma once

#include <cstddef>
#include <span>

#include "core/tokenizer/tokenizer.h"

namespace bllm::cache {

// Axis G: changes with a new cache or context mechanism.
//
// The diff that keys the KV cache on the token sequence. JavaScript sends the
// whole rendered conversation every turn and holds no cache state; this
// function finds how much of it the cache already holds.
//
// A pure function over two token sequences: no GPU, no model, no file. It is
// the most correctness-critical logic in the chat loop, and it is kept apart
// from the cache's storage so it is tested directly.
//
// The runtime truncates the cache to the returned length and prefills the
// rest. When the whole incoming sequence is cached, the runtime still
// recomputes its last token: generation needs that token's logits, and the
// cache holds keys and values, not logits.

[[nodiscard]] std::size_t longest_common_prefix(std::span<const tokenizer::TokenId> cached,
                                                std::span<const tokenizer::TokenId> incoming) noexcept;

}  // namespace bllm::cache

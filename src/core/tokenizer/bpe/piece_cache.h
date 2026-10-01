#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "core/tokenizer/tokenizer.h"

namespace bllm::tokenizer::bpe {

class ByteLevelBpe;

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// The tokens already found for short pieces, so a word that recurs is merged
// once. Text repeats words constantly: the repository's own docs and sources
// split into 87,862 pieces, only 7,006 of them distinct.
//
// The caller owns it and passes it to encode; the encoder keeps no state of
// its own. A cache changes how long encoding takes, never what it returns:
// it holds only results the encoder computed, and a cache that served another
// tokenizer is emptied before it is used again.
//
// Optimization (practice): Hugging Face tokenizers keeps a word cache of the
// same shape — direct-mapped slots of fixed size, so a lookup is one hash and
// one slot, and a collision simply replaces the older word. tiktoken instead
// looks the whole piece up in its vocabulary.
class PieceCache {
public:
    // Measured on that text: 99.3% of pieces are 15 bytes or under, and 99.8%
    // of those encode to at most 3 tokens. Longer pieces are not cached.
    static constexpr std::size_t kMaxBytes = 15;
    static constexpr std::size_t kMaxTokens = 3;

    // `slots`, rounded up to a power of two, each 32 bytes.
    explicit PieceCache(std::size_t slots = 8192);

    // The tokens recorded for `piece`, if it is cached.
    [[nodiscard]] std::optional<std::span<const TokenId>> find(std::string_view piece) const noexcept;

    // Records `piece`'s tokens, replacing whatever held its slot. Pieces or
    // results too long for a slot are not recorded.
    void insert(std::string_view piece, std::span<const TokenId> tokens) noexcept;

    void clear() noexcept;

private:
    friend class ByteLevelBpe;

    // Two to a 64-byte cache line: the 4-byte fields first, so none pads.
    struct Slot {
        std::array<TokenId, kMaxTokens> tokens;
        std::array<char, kMaxBytes> bytes;
        std::uint8_t length;   // 0 when the slot is empty; no piece is empty
        std::uint8_t count;
    };
    static_assert(sizeof(Slot) == 32);

    [[nodiscard]] std::size_t slot_of(std::string_view piece) const noexcept;

    std::vector<Slot> slots_;
    // The tokenizer whose results it holds. Non-owning, and only compared.
    const ByteLevelBpe* owner_ = nullptr;
};

}  // namespace bllm::tokenizer::bpe

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/tokenizer/tokenizer.h"
#include "core/tokenizer/vocabulary.h"

namespace bllm::tokenizer {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// Special tokens found in raw text. The chat template writes them as text, and
// each encodes as its one identifier, never as the characters that spell it.
// Every control and user-defined token is matched as the text stands, before
// any normalization or splitting: no reference marks one to be normalized,
// stripped or matched as a whole word. (SentencePiece matches after spaces
// become "▁", the one change it makes, with its tokens spelled to match:
// sentencepiece_bpe.h.) Text is searched from the left, and at
// each position the longest matching token wins — Hugging Face's
// leftmost-longest match over its added tokens. That matters: Gemma 3's
// special tokens include runs of spaces, each a prefix of the next.

// A run of ordinary text, or one special token, by its byte range.
struct Segment {
    std::uint32_t offset;
    std::uint32_t length;
    std::optional<TokenId> special;   // empty for ordinary text
};

class SpecialTokens {
public:
    // A token and the text that matches it.
    struct Entry {
        std::string text;
        TokenId id;
    };

    // None: every text is ordinary.
    SpecialTokens() = default;

    // Every control and user-defined token with text. It keeps its own copy
    // of their text, so it does not depend on the vocabulary living on.
    explicit SpecialTokens(const Vocabulary& vocabulary);

    // Exactly these, matched by the text each is given. Precondition: no text
    // is empty, and no two are the same.
    explicit SpecialTokens(std::vector<Entry> entries);

    [[nodiscard]] std::size_t size() const noexcept { return tokens_.size(); }

    // Splits `text` into ordinary text and special tokens, in order, covering
    // it exactly. Ordinary runs are never empty. Precondition: `text` is no
    // longer than a Segment can address (4 GiB).
    void segment(std::string_view text, std::vector<Segment>& out) const;

private:
    // Ordered by first byte, then longest first, so the first match at a
    // position is the longest.
    std::vector<Entry> tokens_;
    // The tokens starting with byte b are tokens_[starts_[b], starts_[b + 1]).
    // Optimization (practice): a position whose byte starts no special token
    // costs one check, not one per token; Gemma 3 has 6,414 of them.
    std::array<std::uint32_t, 257> starts_{};
};

}  // namespace bllm::tokenizer

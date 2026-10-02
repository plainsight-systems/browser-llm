#pragma once

#include <array>
#include <vector>

#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/tokenizer/bpe/merge_table.h"
#include "core/tokenizer/special.h"
#include "core/tokenizer/tokenizer.h"
#include "core/tokenizer/vocabulary.h"

namespace bllm::tokenizer::bpe {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// SentencePiece BPE, the algorithm GGUF names "llama", as Gemma 3 uses it.
//
// It is BPE. The file lists no merges, only a score for each token, and
// llama.cpp merges by score: any two adjacent symbols whose text joined is a
// token, highest score first, leftmost on a tie. Hugging Face lists merges
// instead, ranked. They are one algorithm: the merges a vocabulary implies —
// every split of a normal token into two normal tokens, ranked by the score
// of the token they make (sentencepiece_merges.h) — are exactly Hugging
// Face's merges for Gemma 3, 513,511 of them, ranked in the same order of the
// tokens they make. Its other 1,395 make runs of tabs, newlines and spaces,
// which are user-defined tokens, matched in the raw text before anything
// merges. So the merges are derived from the vocabulary once, at load, and
// run through the same merge step as byte-level BPE (merge.h). Encoding a
// rendered prompt:
//
//   1. Special tokens are found in the raw text and become their own
//      identifiers (special.h).
//   2. In the text between them each space becomes "▁" (U+2581), as the
//      vocabulary spells it. Nothing else is normalized, and no space is
//      added before the text.
//   3. Each character becomes its token, or, where the vocabulary has none,
//      one byte token (<0x00> to <0xFF>) for each of its bytes.
//   4. The whole stretch is merged at once; nothing splits it first.
//
// BOS is never added: the chat template writes it as text.
class SentencePieceBpe {
public:
    SentencePieceBpe() = default;

    [[nodiscard]] const Vocabulary& vocabulary() const noexcept { return vocabulary_; }

    // Appends the tokens `text` encodes to. `out` is left untouched if it
    // cannot: text that is not UTF-8, or longer than 4 GiB.
    [[nodiscard]] EncodeError encode(std::string_view text, std::vector<TokenId>& out) const;

private:
    friend LoadResult load_sentencepiece_bpe(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                             SentencePieceBpe& out);

    Vocabulary vocabulary_;
    SpecialTokens special_;
    MergeTable merges_;
    std::array<TokenId, 256> byte_tokens_{};   // <0x00> to <0xFF>
};

// Reads the vocabulary and scores, derives the merges, and finds the byte
// tokens. Refuses a vocabulary with a one-character token that is neither
// normal nor special, which the references would use and this would spell in
// bytes, and a file that asks for a space before the text
// (tokenizer.ggml.add_space_prefix true, or not declared, which llama.cpp
// reads as true): the references place that space differently, llama.cpp
// after every special token and Hugging Face only at the start, and no model
// listed here asks for it. `out` is left untouched unless it succeeds.
[[nodiscard]] LoadResult load_sentencepiece_bpe(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                                SentencePieceBpe& out);

}  // namespace bllm::tokenizer::bpe

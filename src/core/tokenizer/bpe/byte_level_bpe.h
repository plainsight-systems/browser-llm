#pragma once

#include <array>
#include <string_view>
#include <vector>

#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/tokenizer/bpe/merge_table.h"
#include "core/tokenizer/pretokenize.h"
#include "core/tokenizer/special.h"
#include "core/tokenizer/tokenizer.h"
#include "core/tokenizer/vocabulary.h"

namespace bllm::tokenizer::bpe {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// Byte-level BPE, the algorithm GGUF names "gpt2", as Qwen3 and Llama 3 use
// it. Encoding a rendered prompt:
//
//   1. Special tokens are found in the raw text and become their own
//      identifiers (special.h).
//   2. The text between them is normalized as the pre-tokenizer says (NFC
//      for qwen2), then split into pieces (pretokenize.h).
//   3. Each piece's bytes become one token each, by GPT-2's byte map, and are
//      merged (merge.h) — unless the pre-tokenizer takes a piece that is
//      itself a token whole.
//
// BOS is never added: the chat template writes it as text.
class ByteLevelBpe {
public:
    ByteLevelBpe() = default;

    [[nodiscard]] const Vocabulary& vocabulary() const noexcept { return vocabulary_; }

    // Appends the tokens `text` encodes to. `out` is left untouched if it
    // cannot: text that is not UTF-8, or longer than 4 GiB. Precondition:
    // loaded by load_byte_level_bpe; an empty one has no pre-tokenizer.
    [[nodiscard]] EncodeError encode(std::string_view text, std::vector<TokenId>& out) const;

private:
    friend LoadResult load_byte_level_bpe(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                          const PreTokenizer& pretokenizer, ByteLevelBpe& out);

    void encode_piece(std::string_view piece, std::vector<TokenId>& out, std::string& spelled,
                      std::vector<TokenId>& symbols) const;

    Vocabulary vocabulary_;
    SpecialTokens special_;
    MergeTable merges_;
    std::array<TokenId, 256> byte_tokens_{};   // the token for each byte's character
    const PreTokenizer* pretokenizer_ = nullptr;
};

// Reads the vocabulary and merges, and checks every byte has a token. `out`
// is left untouched unless it succeeds.
[[nodiscard]] LoadResult load_byte_level_bpe(gguf::ByteSource& source, const gguf::TensorIndex& index,
                                             const PreTokenizer& pretokenizer, ByteLevelBpe& out);

}  // namespace bllm::tokenizer::bpe

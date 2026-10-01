#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace bllm::tokenizer {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// Split patterns keyed by name — the tokenizer.ggml.pre value — and the
// Unicode category tables they need. The tables are compiled in and pinned to
// one Unicode version (unicode_tables.h), so every browser splits the same
// text the same way; JavaScript's \p{…} classes follow each browser's own
// tables.
//
// The byte-level pre-tokenizers share one pattern, written here by hand
// rather than run through a regex engine; they differ in how many digits one
// piece may hold:
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)   contraction endings, under case folding
//   |[^\r\n\p{L}\p{N}]?\p{L}+      letters, after at most one other character
//   |\p{N}{1,digits}               digits
//   | ?[^\s\p{L}\p{N}]+[\r\n]*     symbols and punctuation, and the line ends
//                                  after them
//   |\s*[\r\n]+                    white space through its last line end
//   |\s+(?!\S)                     white space, but not the space before text
//   |\s+                           any white space left
//
// At each position the first alternative that matches takes its match, and
// splitting resumes after it. Every character matches some alternative, so the
// pieces cover the text exactly.

// What the text goes through before it is split. GGUF does not record a
// normalizer, so it is part of what the pre-tokenizer's name means: "qwen2" is
// the pattern after NFC, as Qwen's tokenizer.json defines it.
enum class Normalization {
    None,
    Nfc,
};

struct PreTokenizer {
    std::string_view name;
    std::uint8_t digits;   // the most digits one piece holds
    Normalization normalization;
};

extern const PreTokenizer kQwen2;      // one digit a piece, after NFC
extern const PreTokenizer kLlamaBpe;   // up to three digits a piece

// A byte range of the input text.
struct Piece {
    std::uint32_t offset;
    std::uint32_t length;
};

enum class SplitError {
    Ok,
    InvalidUtf8,
    TooLong,   // longer than a Piece can address
};

// Splits `text` into the pieces the algorithm encodes independently. The
// pieces are in order and cover `text` exactly. Applies no normalization:
// the caller normalizes first, as the pre-tokenizer's normalization says.
// `out` is left untouched unless the split succeeds.
[[nodiscard]] SplitError split(const PreTokenizer& pretokenizer, std::string_view text,
                               std::vector<Piece>& out);

}  // namespace bllm::tokenizer

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace bllm::tokenizer {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// Split patterns keyed by name — the tokenizer.ggml.pre value — and the
// Unicode category tables they need. The tables are compiled in and pinned to
// the reference tokenizer's Unicode version, so every browser splits the same
// text the same way; JavaScript's \p{…} classes follow each browser's own
// tables.

struct PreTokenizer {
    std::string_view name;
};

// A byte range of the input text.
struct Piece {
    std::uint32_t offset;
    std::uint32_t length;
};

enum class SplitError {
    Ok,
    InvalidUtf8,
};

// Splits `text` into the pieces the algorithm encodes independently. The
// pieces are in order and cover `text` exactly.
[[nodiscard]] SplitError split(const PreTokenizer& pretokenizer, std::string_view text,
                               std::vector<Piece>& out);

}  // namespace bllm::tokenizer

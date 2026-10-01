#pragma once

#include <string>
#include <string_view>

namespace bllm::tokenizer {

// Axis K: changes with a new tokenization algorithm or pre-tokenizer.
//
// Unicode Normalization Form C (UAX #15), for a tokenizer whose reference
// normalizes before splitting, as Qwen's does. The same text written with
// precomposed or with combining characters becomes the same bytes, so it
// becomes the same tokens.
//
// Decompose every character canonically, put each run of combining marks in
// canonical order, then compose each mark with the starter before it unless
// another mark blocks it. The data is unicode_tables.h; conformance is tested
// against Unicode's NormalizationTest.txt for the same version.

// Writes `text` in NFC to `out`. False if `text` is not well-formed UTF-8, in
// which case `out` is left untouched.
[[nodiscard]] bool to_nfc(std::string_view text, std::string& out);

}  // namespace bllm::tokenizer

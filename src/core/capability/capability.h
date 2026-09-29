#pragma once

#include <string_view>

#include "core/gguf/types.h"

namespace bllm::arch { struct Architecture; }
namespace bllm::formats { struct Format; }
namespace bllm::tokenizer { struct Algorithm; struct PreTokenizer; }

namespace bllm::capability {

// Axis I: changes when the harness gains or loses an implementation, and for
// no other reason.
//
// Contract 3. One table from each identifier a GGUF file carries to the
// implementation that runs it. Supported means present here. The gates, format
// dispatch, tokenizer selection and graph selection all find implementations
// through these lookups, so what the picker reports and what the loader runs
// cannot disagree. There is no second list.
//
// Several identifiers may name one implementation (principle 2). A lookup
// returns the implementation, or null if the harness has none; null is not an
// error here, and the gates turn it into a rejection that names what is
// missing.

[[nodiscard]] const arch::Architecture* find_architecture(std::string_view general_architecture) noexcept;

[[nodiscard]] const formats::Format* find_format(gguf::TensorType type) noexcept;

[[nodiscard]] const tokenizer::Algorithm* find_tokenizer(std::string_view tokenizer_model) noexcept;

[[nodiscard]] const tokenizer::PreTokenizer* find_pretokenizer(std::string_view tokenizer_pre) noexcept;

}  // namespace bllm::capability

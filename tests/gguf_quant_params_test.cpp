#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "core/gguf/reader.h"
#include "support/gguf_fixture.h"

// BLLM-002 acceptance criterion 4: every tensor in the index carries its
// quantization parameters, and no consumer can obtain a weight without them.
//
// The hazard this guards is specific. Qwen3-0.6B ties its output projection to
// a Q4_0 token-embedding table that is ~26% of the model. A consumer that gets
// a byte range without the type reads packed nibbles as floats, produces a
// tensor of exactly the right shape, and corrupts every token silently. No
// shape assertion catches it.
//
// What these tests deliberately do NOT assert: that
// bytes_for_elements(t.type, t.element_count) equals t.data_length. The reader
// COMPUTES data_length by calling that function (reader.cpp), so asserting
// they agree re-runs one function on its own inputs and proves nothing. Ground
// truth here comes from the fixture's own definition in
// tools/make_fixture_gguf.py, and from what the type classification implies.
using namespace bllm::gguf;

namespace {

using bllm::testing::load_gguf_fixture;

// The types this harness can interpret. Written out rather than derived from
// bytes_for_elements, so the test states the supported set independently of
// the code that enforces it.
constexpr std::array<TensorType, 3> kInterpretable{
    TensorType::F32, TensorType::F16, TensorType::Q4_0,
};

[[nodiscard]] bool is_interpretable(TensorType t) noexcept {
    return std::find(kInterpretable.begin(), kInterpretable.end(), t) != kInterpretable.end();
}

}  // namespace

TEST_CASE("every entry in the index carries a type the harness can interpret") {
    // Universal, not positional. The parse test pins tensors()[0] and [1] by
    // hand; that passes unchanged if the fixture grows a third tensor whose
    // type nobody can read. This one does not.
    const auto bytes = load_gguf_fixture("valid");
    MemoryByteSource source{bytes};
    Reader reader{source};
    REQUIRE(reader.parse() == ReadError::Ok);

    REQUIRE_FALSE(reader.tensors().empty());
    for (const auto& t : reader.tensors()) {
        CAPTURE(t.name);
        CHECK(is_interpretable(t.type));
        // The quantization classification must follow from the type, not from
        // a separately stored flag that could disagree with it.
        CHECK(t.is_quantized() == (t.type != TensorType::F32 && t.type != TensorType::F16));
        // An entry with no bytes would be interpretable trivially and mean
        // nothing; the fixture's tensors all carry data.
        CHECK(t.data_length > 0);
    }
}

TEST_CASE("the index cannot admit a tensor whose type the harness cannot size") {
    // The negative half of the same claim, and the one the suite was missing.
    // "unknown_tensor_type" (999) fails the RANGE check; "not_block_aligned"
    // fails a Q4_0 shape check. Neither exercises a real ggml type that the
    // range check accepts and this harness still cannot read — which is the
    // path that would otherwise put an uninterpretable entry in the index.
    const auto bytes = load_gguf_fixture("unsupported_tensor_type");
    MemoryByteSource source{bytes};
    Reader reader{source};

    const auto err = reader.parse();
    CHECK(err == ReadError::UnsupportedTensorType);
    CHECK(err != ReadError::UnknownTensorType);   // a different cause, distinctly named
    CHECK_FALSE(to_string(err).empty());
}

TEST_CASE("losing the type is not a silent no-op") {
    // States the failure mode as an assertion rather than as a comment. If a
    // consumer obtained the embedding's byte range without its type and
    // assumed F32, it would read 4 bytes per element. The Q4_0 tensor's real
    // extent is nowhere near that, so the mistake is a memory-safety error and
    // a correctness error at once -- it cannot degrade quietly into slightly
    // wrong numbers.
    const auto bytes = load_gguf_fixture("valid");
    MemoryByteSource source{bytes};
    Reader reader{source};
    REQUIRE(reader.parse() == ReadError::Ok);

    const auto quantized = std::find_if(
        reader.tensors().begin(), reader.tensors().end(),
        [](const TensorEntry& t) { return t.is_quantized(); });
    REQUIRE(quantized != reader.tensors().end());

    const std::uint64_t as_f32 = quantized->element_count * 4;
    CHECK(quantized->data_length < as_f32);
    // Not a near miss: a type-blind reader would run off the end of the file.
    CHECK(quantized->data_offset + as_f32 > source.size());
}

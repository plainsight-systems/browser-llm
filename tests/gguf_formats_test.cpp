#include <doctest/doctest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/gguf/reader.h"
#include "core/quant/q4_0.h"
#include "support/gguf_fixture.h"

// Every tensor in the index carries its format, and the reader knows the
// layout of every format GGUF defines — including formats the harness cannot
// run, which it records rather than rejects.
//
// The hazard this guards is specific. A consumer that gets a byte range
// without its format reads packed blocks as floats, produces a tensor of
// exactly the right shape, and corrupts every value silently. No shape
// assertion catches it.
using namespace bllm::gguf;

namespace {

using bllm::testing::load_gguf_fixture;

TensorIndex read_fixture(const std::vector<std::byte>& bytes) {
    MemoryByteSource source{bytes};
    TensorIndex index;
    REQUIRE(read_index(source, index).error == ReadError::Ok);
    return index;
}

}  // namespace

TEST_CASE("a format the harness cannot run is recorded, not rejected") {
    const auto bytes = load_gguf_fixture("q6_k_tensor");
    const TensorIndex index = read_fixture(bytes);

    REQUIRE(index.tensors().size() == 1);
    const auto& t = index.tensors()[0];
    CHECK(t.type == TensorType::Q6_K);
    CHECK(t.is_quantized());
    // One 256-element super-block of 210 bytes, as the fixture declares.
    CHECK(t.data_length == 210);
}

TEST_CASE("the layout table agrees with the Q4_0 layout stated independently") {
    const FormatLayout* layout = format_layout(TensorType::Q4_0);
    REQUIRE(layout != nullptr);
    CHECK(layout->block_elements == bllm::quant::kQ4_0BlockElements);
    CHECK(layout->block_bytes == bllm::quant::kQ4_0BlockBytes);
}

TEST_CASE("every defined format has a layout, and removed numbers have none") {
    for (const FormatLayout& layout : kFormatLayouts) {
        CAPTURE(layout.name);
        CHECK(format_layout(layout.type) == &layout);
        CHECK(layout.block_elements > 0);
        CHECK(layout.block_bytes > 0);
        CHECK_FALSE(layout.name.empty());
    }
    // Numbers ggml has retired, and the first number past the table.
    for (const std::uint32_t retired : {4u, 5u, 31u, 32u, 33u, 36u, 37u, 38u, 43u}) {
        CAPTURE(retired);
        CHECK(format_layout(static_cast<TensorType>(retired)) == nullptr);
    }
}

TEST_CASE("losing the format is not a silent no-op") {
    // If a consumer obtained the embedding's byte range without its format
    // and assumed F32, it would read 4 bytes per element. The Q4_0 tensor's
    // real extent is nowhere near that, so the mistake is a memory-safety
    // error and a correctness error at once — it cannot degrade quietly into
    // slightly wrong numbers.
    const auto bytes = load_gguf_fixture("valid");
    const TensorIndex index = read_fixture(bytes);

    const auto tensors = index.tensors();
    const auto quantized = std::find_if(tensors.begin(), tensors.end(),
                                        [](const TensorEntry& t) { return t.is_quantized(); });
    REQUIRE(quantized != tensors.end());

    const std::uint64_t as_f32 = quantized->element_count * 4;
    CHECK(quantized->data_length < as_f32);
    // Not a near miss: a format-blind reader would run off the end of the file.
    CHECK(quantized->data_offset + as_f32 > bytes.size());
}

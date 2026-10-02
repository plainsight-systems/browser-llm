// Q4_0's unpack on the GPU, against ggml's dequantize_row_q4_0 mirrored on
// the CPU (tests/support/q4_0_reference.h).

#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/formats/q4_0/q4_0.h"
#include "core/gpu/wgpu_handles.h"
#include "support/acquire.h"
#include "support/q4_0_reference.h"
#include "support/unpack.h"

using namespace bllm;

namespace {

// Blocks with every scale worth testing, ±0, the fp16 subnormal range's ends
// and its largest finite value among them, then random finite ones; the codes
// are random throughout.
std::vector<std::uint8_t> blocks(std::size_t count) {
    const std::uint16_t edges[] = {0x3C00, 0xBC00, 0x0000, 0x8000, 0x0001, 0x03FF, 0x7BFF, 0xFBFF, 0x2E66};
    std::uint32_t state = 0x2545F491u;
    const auto next = [&] { return state = state * 1664525u + 1013904223u; };
    std::vector<std::uint8_t> out;
    for (std::size_t b = 0; b < count; ++b) {
        std::uint16_t d = b < std::size(edges) ? edges[b] : static_cast<std::uint16_t>(next() >> 16);
        if ((d & 0x7C00) == 0x7C00) d &= 0xBBFF;   // never infinity or NaN
        out.push_back(static_cast<std::uint8_t>(d & 0xFF));
        out.push_back(static_cast<std::uint8_t>(d >> 8));
        for (int j = 0; j < 16; ++j) out.push_back(static_cast<std::uint8_t>(next() >> 24));
    }
    return out;
}

}  // namespace

TEST_CASE("Q4_0 unpack decodes every block as ggml does, bit for bit") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = testing::acquire(instance.get());
    // An odd count, so the last scale shares its word with nothing.
    for (const std::size_t count : {std::size_t{1}, std::size_t{9}, std::size_t{257}}) {
        CAPTURE(count);
        const auto stored = blocks(count);
        const auto got = testing::run_unpack(instance.get(), *device, formats::kQ4_0.unpack_wgsl(),
                                             formats::kQ4_0.layout(), std::as_bytes(std::span(stored)),
                                             static_cast<std::uint32_t>(count));
        testing::check_bitwise(got, test::dequantize_q4_0(stored));
    }
}

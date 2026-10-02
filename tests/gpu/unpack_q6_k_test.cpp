// Q6_K's unpack on the GPU, against ggml's dequantize_row_q6_K mirrored on
// the CPU (tests/support/q6_k_reference.h), bit for bit.

#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/formats/q6_k/q6_k.h"
#include "core/gpu/wgpu_handles.h"
#include "support/acquire.h"
#include "support/q6_k_reference.h"
#include "support/unpack.h"

using namespace bllm;

namespace {

// Super-blocks of random codes and scales, -128 and 127 among them, with d
// at +-0, both ends of fp16's subnormals, its largest values, then random
// finite ones.
std::vector<std::uint8_t> blocks(std::size_t count) {
    const std::uint16_t edges[] = {0x3C00, 0xBC00, 0x0000, 0x8000, 0x0001, 0x03FF, 0x7BFF, 0xFBFF, 0x2E66};
    std::uint32_t state = 0x7F4A7C15u;
    const auto next = [&] { return state = state * 1664525u + 1013904223u; };
    std::vector<std::uint8_t> out;
    for (std::size_t b = 0; b < count; ++b) {
        for (int i = 0; i < 192; ++i) out.push_back(static_cast<std::uint8_t>(next() >> 24));   // ql, qh
        for (int i = 0; i < 16; ++i) {
            const auto s = static_cast<std::uint8_t>(next() >> 24);
            out.push_back(i == 0 ? 0x80 : i == 1 ? 0x7F : s);   // scales, -128 and 127 first
        }
        std::uint16_t d = b < std::size(edges) ? edges[b] : static_cast<std::uint16_t>(next() >> 16);
        if ((d & 0x7C00) == 0x7C00) d &= 0xBBFF;   // never infinity or NaN
        out.push_back(static_cast<std::uint8_t>(d & 0xFF));
        out.push_back(static_cast<std::uint8_t>(d >> 8));
    }
    return out;
}

}  // namespace

TEST_CASE("Q6_K unpack decodes every super-block as ggml does, bit for bit") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = testing::acquire(instance.get());
    // An odd count, so the last d shares its word with nothing.
    for (const std::size_t count : {std::size_t{1}, std::size_t{9}, std::size_t{33}}) {
        CAPTURE(count);
        const auto stored = blocks(count);
        const auto got = testing::run_unpack(instance.get(), *device, formats::kQ6_K.unpack_wgsl(),
                                             formats::kQ6_K.layout(), std::as_bytes(std::span(stored)),
                                             static_cast<std::uint32_t>(count * 8));
        testing::check_bitwise(got, test::dequantize_q6_k(stored));
    }
}

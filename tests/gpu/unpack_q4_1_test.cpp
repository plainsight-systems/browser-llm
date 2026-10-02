// Q4_1's unpack on the GPU, against ggml's dequantize_row_q4_1 mirrored on
// the CPU (tests/support/q4_1_reference.h): each weight must equal, bit for
// bit, its multiply-add rounded twice or fused.

#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "core/formats/q4_1/q4_1.h"
#include "core/gpu/wgpu_handles.h"
#include "support/acquire.h"
#include "support/q4_1_reference.h"
#include "support/unpack.h"

using namespace bllm;

namespace {

// Blocks whose scales and minimums include +-0, both ends of fp16's
// subnormals and its largest values, then random finite ones; random codes.
std::vector<std::uint8_t> blocks(std::size_t count) {
    const std::uint16_t edges[] = {0x3C00, 0xBC00, 0x0000, 0x8000, 0x0001, 0x03FF, 0x7BFF, 0xFBFF, 0x2E66};
    std::uint32_t state = 0x9E3779B9u;
    const auto next = [&] { return state = state * 1664525u + 1013904223u; };
    const auto finite = [&](std::size_t i) {
        std::uint16_t h = i < std::size(edges) ? edges[i] : static_cast<std::uint16_t>(next() >> 16);
        if ((h & 0x7C00) == 0x7C00) h &= 0xBBFF;   // never infinity or NaN
        return h;
    };
    std::vector<std::uint8_t> out;
    for (std::size_t b = 0; b < count; ++b) {
        const std::uint16_t d = finite(b);
        // The first nine blocks take every edge as d, and every edge as m,
        // four apart, so no block pairs an edge with itself.
        const std::uint16_t m = finite(b < std::size(edges) ? (b + 4) % std::size(edges) : b);
        for (const std::uint16_t h : {d, m}) {
            out.push_back(static_cast<std::uint8_t>(h & 0xFF));
            out.push_back(static_cast<std::uint8_t>(h >> 8));
        }
        for (int j = 0; j < 16; ++j) out.push_back(static_cast<std::uint8_t>(next() >> 24));
    }
    return out;
}

}  // namespace

TEST_CASE("Q4_1 unpack decodes every block as ggml does, its multiply-add rounded twice or fused") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = testing::acquire(instance.get());
    for (const std::size_t count : {std::size_t{1}, std::size_t{9}, std::size_t{257}}) {
        CAPTURE(count);
        const auto stored = blocks(count);
        const auto got = testing::run_unpack(instance.get(), *device, formats::kQ4_1.unpack_wgsl(),
                                             formats::kQ4_1.layout(), std::as_bytes(std::span(stored)),
                                             static_cast<std::uint32_t>(count));
        if (count >= 9) {
            // Every edge, +-0 among them, is some block's d and some block's m.
            for (const std::uint16_t edge : {0x3C00, 0xBC00, 0x0000, 0x8000, 0x0001, 0x03FF, 0x7BFF, 0xFBFF, 0x2E66}) {
                bool as_d = false;
                bool as_m = false;
                for (std::size_t b = 0; b < count; ++b) {
                    as_d |= (stored[b * 20] | (stored[b * 20 + 1] << 8)) == edge;
                    as_m |= (stored[b * 20 + 2] | (stored[b * 20 + 3] << 8)) == edge;
                }
                CAPTURE(edge);
                CHECK(as_d);
                CHECK(as_m);
            }
        }
        const auto want = test::dequantize_q4_1(stored);
        REQUIRE(got.size() == want.rounded.size());
        std::size_t differing = 0;
        std::size_t fused = 0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            const bool as_rounded = testing::same_weight(got[i], want.rounded[i]);
            const bool as_fused = testing::same_weight(got[i], want.fused[i]);
            if (!as_rounded && as_fused) ++fused;
            if (!as_rounded && !as_fused && differing++ < 8) {
                CAPTURE(i);
                CHECK(got[i] == want.rounded[i]);
            }
        }
        CHECK(differing == 0);
        MESSAGE(fused << " of " << got.size() << " weights matched only the fused evaluation");
    }
}

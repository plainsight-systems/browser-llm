// Q6_K's repack against the stored block it comes from: every weight's six
// bits, every scale and d, found where q6_k.h says the repacked block holds
// them, from where ggml's block_q6_K stores them. The GPU test of unpack
// (gpu/unpack_q6_k_test.cpp) checks the decode; this checks the layout, with
// no device.

#include <doctest/doctest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/formats/device_layout.h"

using namespace bllm;

namespace {

constexpr std::size_t kBlock = 210;

// Weight l of group k, stored: ggml's dequantize_row_q6_K indexing.
unsigned stored_q(const std::byte* block, unsigned k, unsigned l) {
    const unsigned h = k / 4;
    const unsigned r = k % 4;
    const auto ql = static_cast<unsigned>(block[64 * h + 32 * (r % 2) + l]);
    const auto qh = static_cast<unsigned>(block[128 + 32 * h + l]);
    return ((ql >> (4 * (r / 2))) & 0xF) | (((qh >> (2 * r)) & 3) << 4);
}

// Weight l of group k, repacked: q6_k.h's layout, in its streams.
unsigned repacked_q(const std::byte* lo, const std::byte* hi, unsigned k, unsigned l) {
    const auto nibbles = static_cast<unsigned>(lo[16 * k + l % 16]);
    const auto pairs = static_cast<unsigned>(hi[8 * k + l % 8]);
    return ((nibbles >> (4 * (l / 16))) & 0xF) | (((pairs >> (2 * (l / 8))) & 3) << 4);
}

}  // namespace

TEST_CASE("Q6_K's repack puts every weight, scale and d where its layout says") {
    // Three blocks, so each stream's place advances from one to the next.
    constexpr std::size_t kBlocks = 3;
    std::vector<std::byte> stored(kBlocks * kBlock);
    std::uint32_t state = 0x2545F491u;
    for (std::byte& b : stored) {
        state = state * 1664525u + 1013904223u;
        b = static_cast<std::byte>(state >> 24);
    }
    std::vector<std::byte> lo(kBlocks * 128), hi(kBlocks * 64), scales(kBlocks * 16), d(kBlocks * 2);
    const auto& layout = formats::kQ6_KLayout;
    REQUIRE(layout.repack != nullptr);
    layout.repack(stored, {lo.data(), hi.data(), scales.data(), d.data()});

    for (std::size_t b = 0; b < kBlocks; ++b) {
        const std::byte* block = &stored[b * kBlock];
        for (unsigned k = 0; k < 8; ++k) {
            for (unsigned l = 0; l < 32; ++l) {
                CAPTURE(b);
                CAPTURE(k);
                CAPTURE(l);
                REQUIRE(repacked_q(&lo[b * 128], &hi[b * 64], k, l) == stored_q(block, k, l));
            }
        }
        for (unsigned i = 0; i < 16; ++i) CHECK(scales[b * 16 + i] == block[192 + i]);
        CHECK(d[b * 2] == block[208]);
        CHECK(d[b * 2 + 1] == block[209]);
    }
}

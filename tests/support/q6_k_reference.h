#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/quant/q4_0.h"

// CPU reference dequantizer for Q6_K — TEST ORACLE ONLY, as
// q4_0_reference.h is. Mirrors ggml's dequantize_row_q6_K line for line: a
// 256-weight super-block of 128 bytes of low nibbles (ql), 64 of high bit
// pairs (qh), 16 signed 8-bit scales and an fp16 d, decoded as two halves of
// 128 weights, each weight d * scale * q in that order, q in -32 .. 31.
namespace bllm::test {

[[nodiscard]] inline std::vector<float> dequantize_q6_k(std::span<const std::uint8_t> data) {
    constexpr std::size_t kBlockBytes = 210;
    const std::size_t blocks = data.size() / kBlockBytes;
    std::vector<float> out(blocks * 256);
    float* y = out.data();
    for (std::size_t i = 0; i < blocks; ++i) {
        const std::uint8_t* block = data.data() + i * kBlockBytes;
        const std::uint8_t* ql = block;
        const std::uint8_t* qh = block + 128;
        const auto* sc = reinterpret_cast<const std::int8_t*>(block + 192);
        const float d = quant::fp16_to_fp32(static_cast<std::uint16_t>(block[208] | (block[209] << 8)));
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const auto q1 = static_cast<std::int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const auto q2 = static_cast<std::int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const auto q3 = static_cast<std::int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const auto q4 = static_cast<std::int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
                y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
                y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
                y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return out;
}

}  // namespace bllm::test

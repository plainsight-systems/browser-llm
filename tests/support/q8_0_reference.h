#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/quant/q4_0.h"

// CPU reference dequantizer for Q8_0 — TEST ORACLE ONLY, as
// q4_0_reference.h is. Mirrors ggml's dequantize_row_q8_0: a block is an fp16
// d and 32 signed 8-bit codes, and weight j is qs[j] * d.
namespace bllm::test {

[[nodiscard]] inline std::vector<float> dequantize_q8_0(std::span<const std::uint8_t> data) {
    constexpr std::size_t kBlockBytes = 34;
    const std::size_t blocks = data.size() / kBlockBytes;
    std::vector<float> out(blocks * 32);
    for (std::size_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data.data() + b * kBlockBytes;
        const float d = quant::fp16_to_fp32(static_cast<std::uint16_t>(block[0] | (block[1] << 8)));
        for (std::size_t j = 0; j < 32; ++j) {
            out[b * 32 + j] = static_cast<float>(static_cast<std::int8_t>(block[2 + j])) * d;
        }
    }
    return out;
}

}  // namespace bllm::test

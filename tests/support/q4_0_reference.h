#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/quant/q4_0.h"

// CPU reference dequantizer — TEST ORACLE ONLY.
//
// This is deliberately in tests/, not src/core/. Production never materialises
// a dequantized weight: the unpack happens inside the matmul kernel, which is
// the entire reason quantization saves bandwidth. A dequantizer sitting in core
// would be an invitation to call it.
//
// Written to be obviously correct rather than fast, so it can serve as the
// oracle for the GPU kernel. It mirrors ggml's dequantize_row_q4_0 structure
// exactly, because agreeing with the format's reference implementation is the
// point.
namespace bllm::test {

// Dequantizes `blocks` worth of Q4_0 data. `data` must be exactly
// block_count * 18 bytes. Returns block_count * 32 floats.
[[nodiscard]] inline std::vector<float> dequantize_q4_0(
    std::span<const std::uint8_t> data) {
    namespace q = bllm::quant;

    const std::size_t block_count = data.size() / q::kQ4_0BlockBytes;
    std::vector<float> out(block_count * q::kQ4_0BlockElements);

    for (std::size_t b = 0; b < block_count; ++b) {
        const std::uint8_t* block = data.data() + b * q::kQ4_0BlockBytes;

        // Scale is little-endian fp16 in the first two bytes.
        const auto raw = static_cast<std::uint16_t>(
            static_cast<std::uint16_t>(block[0]) |
            (static_cast<std::uint16_t>(block[1]) << 8));
        const float d = q::fp16_to_fp32(raw);

        const std::uint8_t* qs = block + q::kQ4_0ScaleBytes;
        for (std::size_t j = 0; j < q::kQ4_0NibbleBytes; ++j) {
            const int lo = static_cast<int>(qs[j] & 0x0F) - q::kQ4_0ZeroPoint;
            const int hi = static_cast<int>(qs[j] >> 4) - q::kQ4_0ZeroPoint;

            // Byte j carries elements j and j + 16 — not a sequential pair.
            out[b * q::kQ4_0BlockElements + j] = static_cast<float>(lo) * d;
            out[b * q::kQ4_0BlockElements + j + q::kQ4_0NibbleBytes] =
                static_cast<float>(hi) * d;
        }
    }
    return out;
}

}  // namespace bllm::test

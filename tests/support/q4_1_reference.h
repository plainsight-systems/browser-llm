#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/quant/q4_0.h"

// CPU reference dequantizer for Q4_1 — TEST ORACLE ONLY, as
// q4_0_reference.h is. It mirrors ggml's dequantize_row_q4_1: weight j is
// (qs[j] & 0xF) * d + m and weight j + 16 is (qs[j] >> 4) * d + m.
//
// That multiply-add may be evaluated two ways, both correct: rounded after
// the multiply and again after the add, or fused into one fma, rounded once.
// C++ compilers fuse it or not by platform and flags, and WGSL leaves it to
// the implementation, so ggml itself gives either. The reference returns
// both, computed explicitly, and a decoded weight must equal one of them
// exactly.
namespace bllm::test {

struct Q4_1Reference {
    std::vector<float> rounded;   // the multiply and the add each rounded
    std::vector<float> fused;     // one fma
};

[[nodiscard]] inline Q4_1Reference dequantize_q4_1(std::span<const std::uint8_t> data) {
    constexpr std::size_t kBlockBytes = 20;
    constexpr std::size_t kHalf = 16;
    const std::size_t blocks = data.size() / kBlockBytes;
    Q4_1Reference out{std::vector<float>(blocks * 32), std::vector<float>(blocks * 32)};
    const auto half = [](const std::uint8_t* p) {
        return quant::fp16_to_fp32(static_cast<std::uint16_t>(p[0] | (p[1] << 8)));
    };
    for (std::size_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data.data() + b * kBlockBytes;
        const float d = half(block);
        const float m = half(block + 2);
        const std::uint8_t* qs = block + 4;
        for (std::size_t j = 0; j < kHalf; ++j) {
            const auto lo = static_cast<float>(qs[j] & 0x0F);
            const auto hi = static_cast<float>(qs[j] >> 4);
            // Rounded twice only because the GPU tests are compiled with
            // -ffp-contract=off (tests/gpu/CMakeLists.txt): GCC's default
            // would fuse these across statements.
            const float lo_d = lo * d;
            const float hi_d = hi * d;
            out.rounded[b * 32 + j] = lo_d + m;
            out.rounded[b * 32 + j + kHalf] = hi_d + m;
            out.fused[b * 32 + j] = std::fma(lo, d, m);
            out.fused[b * 32 + j + kHalf] = std::fma(hi, d, m);
        }
    }
    return out;
}

}  // namespace bllm::test

#pragma once

#include <cstddef>
#include <cstdint>

// Q4_0 block layout, as defined by the GGUF/ggml format.
//
// This header describes the on-disk layout only. It deliberately does NOT
// dequantize: production never materialises a dequantized weight, on CPU or
// GPU — the unpack happens inside the matmul kernel. A CPU dequantizer exists
// solely as a test oracle and lives in tests/, where it cannot be mistaken for
// a production path.
namespace bllm::quant {

// Weights per block. Fixed by the format.
inline constexpr std::size_t kQ4_0BlockElements = 32;

// One fp16 scale followed by 16 bytes of packed nibbles.
inline constexpr std::size_t kQ4_0ScaleBytes = 2;
inline constexpr std::size_t kQ4_0NibbleBytes = kQ4_0BlockElements / 2;
inline constexpr std::size_t kQ4_0BlockBytes = kQ4_0ScaleBytes + kQ4_0NibbleBytes;
static_assert(kQ4_0BlockBytes == 18);

// Quantized values are stored biased; the true value is (nibble - 8) * scale.
inline constexpr int kQ4_0ZeroPoint = 8;

// Blocks needed to hold `elements` weights. Q4_0 tensors are always a whole
// number of blocks; a remainder means the tensor is not Q4_0-shaped.
[[nodiscard]] constexpr bool is_block_aligned(std::size_t elements) noexcept {
    return elements % kQ4_0BlockElements == 0;
}

[[nodiscard]] constexpr std::size_t block_count(std::size_t elements) noexcept {
    return elements / kQ4_0BlockElements;
}

[[nodiscard]] constexpr std::size_t bytes_for(std::size_t elements) noexcept {
    return block_count(elements) * kQ4_0BlockBytes;
}

// Which byte within a block holds a given element's nibble, and whether it is
// the high or low half.
//
// The packing is NOT sequential pairs. Byte j holds element j in its low
// nibble and element j + 16 in its high nibble, so the two halves of a block
// are interleaved across the same 16 bytes. Getting this backwards produces a
// correctly-shaped tensor whose values are transposed within every block —
// the kind of defect that survives a shape assertion.
struct NibbleLocation {
    std::size_t byte_index;  // 0..15, relative to the start of the nibbles
    bool high;               // true -> upper 4 bits
};

[[nodiscard]] constexpr NibbleLocation locate_nibble(std::size_t element_in_block) noexcept {
    const bool high = element_in_block >= kQ4_0NibbleBytes;
    return NibbleLocation{high ? element_in_block - kQ4_0NibbleBytes : element_in_block,
                          high};
}

// IEEE 754 binary16 -> binary32.
//
// Written by hand rather than using _Float16 so the conversion is exactly
// specified and testable, including subnormals, infinities and NaN. Every
// binary16 value is exactly representable in binary32, so this is lossless and
// has one correct answer per input.
[[nodiscard]] constexpr float fp16_to_fp32(std::uint16_t h) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t exponent = (h >> 10) & 0x1Fu;
    const std::uint32_t mantissa = h & 0x03FFu;

    std::uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;                       // +/- zero
        } else {
            // Subnormal: renormalise into binary32's exponent range.
            std::uint32_t m = mantissa;
            std::uint32_t e = 0;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                ++e;
            }
            m &= 0x03FFu;
            bits = sign | ((127u - 15u - e + 1u) << 23) | (m << 13);
        }
    } else if (exponent == 0x1Fu) {
        bits = sign | 0x7F800000u | (mantissa << 13);   // inf / NaN
    } else {
        bits = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }

    // std::bit_cast would be cleaner but is not constexpr-friendly across all
    // our targets without <bit>; this is the same operation, spelled out.
    return __builtin_bit_cast(float, bits);
}

}  // namespace bllm::quant

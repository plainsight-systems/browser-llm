#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/quant/q4_0.h"
#include "support/q4_0_reference.h"

using namespace bllm::quant;
using bllm::test::dequantize_q4_0;

TEST_CASE("block geometry matches the format definition") {
    CHECK(kQ4_0BlockElements == 32);
    CHECK(kQ4_0BlockBytes == 18);          // 2-byte scale + 16 bytes of nibbles
    CHECK(bytes_for(32) == 18);
    CHECK(bytes_for(1024) == 32 * 18);
    CHECK(block_count(1024) == 32);
    CHECK(is_block_aligned(1024));
    CHECK_FALSE(is_block_aligned(1000));   // not a whole number of blocks
    CHECK_FALSE(is_block_aligned(1));
}

TEST_CASE("nibble packing interleaves the block halves, it does not pair") {
    // Byte j carries element j and element j+16. Getting this backwards
    // transposes values within every block while preserving the shape.
    CHECK(locate_nibble(0).byte_index == 0);
    CHECK_FALSE(locate_nibble(0).high);

    CHECK(locate_nibble(15).byte_index == 15);
    CHECK_FALSE(locate_nibble(15).high);

    CHECK(locate_nibble(16).byte_index == 0);      // NOT byte 8
    CHECK(locate_nibble(16).high);

    CHECK(locate_nibble(31).byte_index == 15);
    CHECK(locate_nibble(31).high);
}

TEST_CASE("fp16 conversion is exact across the classes that matter") {
    CHECK(fp16_to_fp32(0x0000) == 0.0f);
    CHECK(fp16_to_fp32(0x8000) == 0.0f);
    CHECK(std::signbit(fp16_to_fp32(0x8000)));       // negative zero preserved
    CHECK(fp16_to_fp32(0x3C00) == 1.0f);
    CHECK(fp16_to_fp32(0xBC00) == -1.0f);
    CHECK(fp16_to_fp32(0x4000) == 2.0f);
    CHECK(fp16_to_fp32(0x3800) == 0.5f);
    CHECK(fp16_to_fp32(0x7BFF) == 65504.0f);         // largest finite binary16

    // Smallest positive subnormal: exactly 2^-24. Written as a hex float so
    // the expectation is exact — a decimal literal rounds and the comparison
    // then tests the literal, not the conversion.
    CHECK(fp16_to_fp32(0x0001) == 0x1p-24f);
    // Second subnormal is exactly twice the first; the renormalise loop must
    // shift by one less.
    CHECK(fp16_to_fp32(0x0002) == 0x1p-23f);
    // Largest subnormal, just below the smallest normal.
    CHECK(fp16_to_fp32(0x03FF) < fp16_to_fp32(0x0400));

    CHECK(std::isinf(fp16_to_fp32(0x7C00)));
    CHECK(std::isinf(fp16_to_fp32(0xFC00)));
    CHECK(std::signbit(fp16_to_fp32(0xFC00)));
    CHECK(std::isnan(fp16_to_fp32(0x7E00)));
}

namespace {

// Builds one block with a known scale and known nibbles.
std::vector<std::uint8_t> make_block(std::uint16_t scale_fp16,
                                     const std::vector<std::uint8_t>& nibbles) {
    REQUIRE(nibbles.size() == kQ4_0BlockElements);
    std::vector<std::uint8_t> block(kQ4_0BlockBytes, 0);
    block[0] = static_cast<std::uint8_t>(scale_fp16 & 0xFF);
    block[1] = static_cast<std::uint8_t>(scale_fp16 >> 8);
    for (std::size_t j = 0; j < kQ4_0NibbleBytes; ++j) {
        // element j in the low nibble, element j+16 in the high nibble
        block[kQ4_0ScaleBytes + j] =
            static_cast<std::uint8_t>((nibbles[j] & 0x0F) |
                                      ((nibbles[j + kQ4_0NibbleBytes] & 0x0F) << 4));
    }
    return block;
}

}  // namespace

TEST_CASE("dequantization is exact against independently computed values") {
    // scale = 1.0, so the expected output is simply (nibble - 8).
    std::vector<std::uint8_t> nibbles(kQ4_0BlockElements);
    for (std::size_t i = 0; i < kQ4_0BlockElements; ++i) {
        nibbles[i] = static_cast<std::uint8_t>(i % 16);
    }
    const auto block = make_block(0x3C00, nibbles);
    const auto out = dequantize_q4_0(block);

    REQUIRE(out.size() == kQ4_0BlockElements);
    for (std::size_t i = 0; i < kQ4_0BlockElements; ++i) {
        const float expected = static_cast<float>(static_cast<int>(i % 16) - 8);
        // Bit-exact: a single multiply by 1.0 has exactly one correct answer.
        CHECK(out[i] == expected);
    }
}

TEST_CASE("the zero point is 8, so nibble 8 dequantizes to exactly zero") {
    std::vector<std::uint8_t> nibbles(kQ4_0BlockElements, 8);
    const auto out = dequantize_q4_0(make_block(0x4900, nibbles));  // scale 10.0
    for (const float v : out) {
        CHECK(v == 0.0f);
    }
}

TEST_CASE("nibble extremes map to the full signed range") {
    std::vector<std::uint8_t> nibbles(kQ4_0BlockElements, 0);
    nibbles[0] = 0;    // -8
    nibbles[1] = 15;   // +7
    const auto out = dequantize_q4_0(make_block(0x3C00, nibbles));
    CHECK(out[0] == -8.0f);
    CHECK(out[1] == 7.0f);
}

TEST_CASE("element 16 comes from the high nibble of byte 0, not byte 8") {
    // The test that catches a sequential-pair implementation. Only element 16
    // is non-zero; a wrong unpack puts the value at index 1 or 8.
    std::vector<std::uint8_t> nibbles(kQ4_0BlockElements, 8);  // all -> 0.0
    nibbles[16] = 15;                                          // -> +7
    const auto out = dequantize_q4_0(make_block(0x3C00, nibbles));

    CHECK(out[16] == 7.0f);
    CHECK(out[0] == 0.0f);
    CHECK(out[1] == 0.0f);
    CHECK(out[8] == 0.0f);
}

TEST_CASE("scale is applied per block, not per tensor") {
    std::vector<std::uint8_t> a(kQ4_0BlockElements, 9);   // nibble 9 -> +1
    std::vector<std::uint8_t> b(kQ4_0BlockElements, 9);
    auto data = make_block(0x3C00, a);                   // scale 1.0
    const auto second = make_block(0x4000, b);           // scale 2.0
    data.insert(data.end(), second.begin(), second.end());

    const auto out = dequantize_q4_0(data);
    REQUIRE(out.size() == 2 * kQ4_0BlockElements);
    CHECK(out[0] == 1.0f);                               // first block
    CHECK(out[kQ4_0BlockElements] == 2.0f);              // second block
}

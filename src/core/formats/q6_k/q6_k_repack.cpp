#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

#include "core/formats/device_layout.h"

// Q6_K's repack (device_layout.h), from the stored super-block ggml's
// block_q6_K defines to the one q6_k.h describes and q6_k.wgsl reads.
//
// Stored, weight l of run r in half h — group k = 4h + r of the block — has
// its low four bits as nibble r / 2 of ql byte 64h + 32(r % 2) + l, and its
// high two bits as bits 2r, 2r + 1 of qh byte 32h + l. Repacked, its low
// nibble is in lo byte 16k + l % 16, low half for l < 16, high half after;
// its high pair in hi byte 8k + l % 8, at bits 2(l / 8). Scales and d are
// copied as stored.
//
// Each step moves eight bytes, eight weights, at once: a shift and a mask
// pick one nibble or one bit pair out of every byte of a word, and a second
// shift places it. A half's 32 qh bytes are loaded once for its four groups.
// Optimization (practice): no loop runs per weight or per field, so a block
// costs a few dozen word operations, not 256 bit extractions. make bench
// (bench/piece_writer_bench.cpp), 360 MiB, native release, Apple M3 Max:
// 13.4 ms (28.1 GB/s), against 14.0 ms for the stored fields gathered as they
// were, and 6.0 ms for memcpy of the same bytes; one pass writes all four
// streams.

namespace bllm::formats::detail {
namespace {

static_assert(std::endian::native == std::endian::little,
              "the repack reads words of bytes in file order; wasm and the native targets are little-endian");

std::uint64_t load(const std::byte* p) {
    std::uint64_t v = 0;
    std::memcpy(&v, p, sizeof v);
    return v;
}

void store(std::byte* p, std::uint64_t v) { std::memcpy(p, &v, sizeof v); }

constexpr std::uint64_t kNibbles = 0x0F0F0F0F0F0F0F0Full;   // the low nibble of every byte
constexpr std::uint64_t kPairs = 0x0303030303030303ull;     // the low bit pair of every byte

}  // namespace

void repack_q6_k(std::span<const std::byte> blocks, const std::array<std::byte*, kMaxStreams>& streams) {
    std::byte* lo = streams[0];
    std::byte* hi = streams[1];
    std::byte* scales = streams[2];
    std::byte* d = streams[3];
    for (std::size_t b = 0; b < blocks.size(); b += 210, lo += 128, hi += 64, scales += 16, d += 2) {
        const std::byte* block = blocks.data() + b;
        for (unsigned h = 0; h < 2; ++h) {
            const std::byte* ql = block + 64 * h;
            const std::byte* qh = block + 128 + 32 * h;
            const std::uint64_t pairs[4] = {load(qh), load(qh + 8), load(qh + 16), load(qh + 24)};
            for (unsigned r = 0; r < 4; ++r) {
                const unsigned k = 4 * h + r;
                const std::byte* run = ql + 32 * (r % 2);
                const unsigned nibble = 4 * (r / 2);
                // lo byte j: weight j's nibble, then weight j + 16's.
                for (unsigned part = 0; part < 2; ++part) {
                    const std::uint64_t first = (load(run + 8 * part) >> nibble) & kNibbles;
                    const std::uint64_t second = (load(run + 16 + 8 * part) >> nibble) & kNibbles;
                    store(lo + 16 * k + 8 * part, first | (second << 4));
                }
                // hi byte m: the pairs of weights m, m + 8, m + 16 and m + 24.
                std::uint64_t packed = 0;
                for (unsigned t = 0; t < 4; ++t) packed |= ((pairs[t] >> (2 * r)) & kPairs) << (2 * t);
                store(hi + 8 * k, packed);
            }
        }
        std::memcpy(scales, block + 192, 16);
        std::memcpy(d, block + 208, 2);
    }
}

}  // namespace bllm::formats::detail

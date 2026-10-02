#pragma once

#include <cstdint>
#include <span>

#include "core/gguf/types.h"

namespace bllm::formats {

// Axis B: changes with a new weight format.
//
// How a format's blocks lie on the device. A file stores each block whole:
// a Q4_0 block is a 2-byte scale and 16 bytes of nibbles, 18 bytes in all.
// On the device a piece of a weight holds its blocks' fields as streams
// instead: every block's nibbles, in block order, then every block's scales.
// Upload writes a piece that way (residency/piece_writer.h) and a format's
// unpack reads it that way.
//
// Optimization (browser): WGSL reads a storage buffer in aligned 32-bit
// words. An 18-byte block straddles them, so reading one stored whole takes
// two loads and a shift for every field that crosses a word; as streams,
// every field of every block starts on the alignment its width needs.
// Optimization (practice): adjacent invocations read adjacent blocks, and
// with each field a stream they touch adjacent addresses, the access GPU.2
// asks for. llama.cpp's OpenCL backend converts Q4_0 and Q8_0 weights to the
// same struct of arrays as it uploads them (GGML_OPENCL_SOA_Q).
//
// A layout's streams cover its block exactly, each byte in one stream. At most
// the last stream's width may leave a stream unaligned, so a piece of n blocks
// takes exactly its stored bytes rounded up to 4 — the length the residency
// plan already gives it — whatever n is. That is checked at compile time
// below, for every layout listed (P.5).
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     P.5    Prefer compile-time checking to run-time checking — each layout's
//            promise is a static_assert, not a test.
//     Con.5  Use constexpr for values that can be computed at compile time —
//            the layouts are a constexpr table.
//   C++ performance guidelines
//     GPU.2  Shape data for coalesced lane access before tuning the kernel —
//            the streams.
//     EMB.6  Push computation to compile time with constexpr and consteval —
//            the check is consteval, so it cannot drift to run time.

// One field of a block: where it lies in the stored block, and how wide it is.
struct Stream {
    std::uint16_t offset;
    std::uint16_t width;
};

struct DeviceLayout {
    gguf::TensorType type;
    std::uint32_t block_bytes;
    std::span<const Stream> streams;   // in device order
};

// The layouts of the formats the listed models use: F32 for norms, Q4_0 and
// Q4_1 for most weights, Q8_0 and Q6_K for embeddings. ggml's block structs
// (ggml-common.h) give each field's place in the stored block.
namespace detail {
inline constexpr Stream kF32[] = {{0, 4}};
inline constexpr Stream kQ4_0[] = {{2, 16}, {0, 2}};                     // qs, then d
inline constexpr Stream kQ4_1[] = {{4, 16}, {0, 4}};                     // qs, then d and m
inline constexpr Stream kQ8_0[] = {{2, 32}, {0, 2}};                     // qs, then d
inline constexpr Stream kQ6_K[] = {{0, 128}, {128, 64}, {192, 16}, {208, 2}};   // ql, qh, scales, d
}  // namespace detail

inline constexpr DeviceLayout kDeviceLayouts[] = {
    {gguf::TensorType::F32, 4, detail::kF32},
    {gguf::TensorType::Q4_0, 18, detail::kQ4_0},
    {gguf::TensorType::Q4_1, 20, detail::kQ4_1},
    {gguf::TensorType::Q8_0, 34, detail::kQ8_0},
    {gguf::TensorType::Q6_K, 210, detail::kQ6_K},
};

// The layout for `type`, or null if none is listed.
[[nodiscard]] constexpr const DeviceLayout* device_layout(gguf::TensorType type) noexcept {
    for (const DeviceLayout& layout : kDeviceLayouts) {
        if (layout.type == type) return &layout;
    }
    return nullptr;
}

// Whether `layout` keeps the promise above: its streams cover the block, its
// block size is the file format's, and only its last stream may be unaligned.
[[nodiscard]] consteval bool keeps_its_promise(const DeviceLayout& layout) {
    std::uint32_t covered = 0;
    bool seen[256] = {};
    for (std::size_t i = 0; i < layout.streams.size(); ++i) {
        const Stream s = layout.streams[i];
        if (i + 1 < layout.streams.size() && s.width % 4 != 0) return false;
        for (std::uint32_t b = s.offset; b < std::uint32_t{s.offset} + s.width; ++b) {
            if (b >= layout.block_bytes || b >= 256 || seen[b]) return false;
            seen[b] = true;
        }
        covered += s.width;
    }
    const gguf::FormatLayout* stored = gguf::format_layout(layout.type);
    return covered == layout.block_bytes && stored != nullptr && stored->block_bytes == layout.block_bytes;
}

static_assert([] {
    for (const DeviceLayout& layout : kDeviceLayouts) {
        if (!keeps_its_promise(layout)) return false;
    }
    return true;
}());

}  // namespace bllm::formats

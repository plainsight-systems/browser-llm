// F32's unpack on the GPU: every float comes back as stored, bit for bit.

#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "core/formats/f32/f32.h"
#include "core/gpu/wgpu_handles.h"
#include "support/acquire.h"
#include "support/unpack.h"

using namespace bllm;

TEST_CASE("F32 unpack returns every float as stored, -0 and subnormals included") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = testing::acquire(instance.get());
    // +-0, the smallest and largest subnormals, the smallest normal, the
    // largest finite, then random finite values.
    const std::uint32_t edges[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x807FFFFFu,
                                   0x00800000u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x3F800000u};
    for (const std::uint32_t groups : {1u, 3u, 33u}) {
        CAPTURE(groups);
        std::vector<std::uint32_t> words(std::size_t{groups} * 32);
        std::uint32_t state = 0x6C078965u;
        for (std::size_t i = 0; i < words.size(); ++i) {
            state = state * 1664525u + 1013904223u;
            std::uint32_t w = i < std::size(edges) ? edges[i] : state;
            if ((w & 0x7F800000u) == 0x7F800000u) w &= 0xBFFFFFFFu;   // never infinity or NaN
            words[i] = w;
        }
        std::vector<float> stored(words.size());
        std::memcpy(stored.data(), words.data(), words.size() * 4);
        const auto got = testing::run_unpack(instance.get(), *device, formats::kF32.unpack_wgsl(),
                                             formats::kF32.layout(), std::as_bytes(std::span(stored)), groups);
        testing::check_bitwise(got, stored);
    }
}

// Upload's CPU stage, the piece writer, on each quantized layout: make bench.
//
// For each layout a listed model uses, it rearranges a 360 MiB piece of
// synthetic blocks, about a listed model's weights, in 16 MiB chunks as
// upload does, and prints the time and throughput beside memcpy of the same
// bytes, the ceiling for any stage that reads and writes every byte once
// (GDSA.6). Every figure is printed with its conditions (WASM.11): the build,
// how many runs were timed and how many warm-up runs were discarded, and the
// spread around the median.
//
// It checks that the writes cover the piece's device bytes exactly before
// it reports any time.
//
//     charlotte_bench_piece_writer

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

#include "core/formats/device_layout.h"
#include "core/residency/piece_writer.h"
#include "core/residency/routes.h"

using namespace bllm;

namespace {

constexpr int kWarmUp = 2;   // runs discarded
constexpr int kRuns = 11;    // runs timed; the median is the 6th
constexpr std::size_t kChunk = 16u << 20;
constexpr std::uint64_t kPieceBytes = 360ull << 20;

struct Timing {
    double median_ms;
    double p10_ms;
    double p90_ms;
};

template <typename Run>
Timing measure(Run run) {
    for (int i = 0; i < kWarmUp; ++i) run();
    std::array<double, kRuns> ms{};
    for (double& m : ms) {
        const auto start = std::chrono::steady_clock::now();
        run();
        m = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    std::sort(ms.begin(), ms.end());
    return {ms[kRuns / 2], ms[kRuns / 10], ms[kRuns * 9 / 10]};
}

void report(const char* what, std::uint64_t bytes, const Timing& t) {
    std::printf("  %-26s %7.2f ms (%.2f-%.2f)  %6.2f GB/s\n", what, t.median_ms, t.p10_ms, t.p90_ms,
                static_cast<double>(bytes) / t.median_ms / 1e6);
}

}  // namespace

int main() {
    std::printf("piece writer, %d runs timed after %d discarded, median (p10-p90), %s build\n", kRuns, kWarmUp,
#ifdef NDEBUG
                "release"
#else
                "debug"
#endif
    );
    std::vector<std::byte> copy_to(kPieceBytes);
    for (const formats::DeviceLayout* layout :
         {&formats::kQ4_0Layout, &formats::kQ4_1Layout, &formats::kQ8_0Layout, &formats::kQ6_KLayout}) {
        const std::uint64_t blocks = kPieceBytes / layout->block_bytes;
        const std::uint64_t bytes = blocks * layout->block_bytes;
        std::vector<std::byte> file(bytes);
        for (std::size_t i = 0; i < file.size(); ++i) file[i] = static_cast<std::byte>(i * 131 + 7);
        const std::uint64_t length = (bytes + 3) / 4 * 4;
        const residency::Route route{gguf::TensorId{0}, 0, blocks, layout, residency::BufferIndex{0}, 0, length};

        std::vector<residency::Write> writes;
        const auto rearrange = [&] {
            residency::PieceWriter writer(std::span(&route, 1), kChunk);
            std::uint64_t written = 0;
            for (std::uint64_t at = 0; at < bytes; at += kChunk) {
                writes.clear();
                const auto n = std::min<std::uint64_t>(kChunk, bytes - at);
                if (writer.accept(at, std::span(file).subspan(at, n), writes) != residency::WriteError::Ok) return 0ull;
                for (const auto& w : writes) written += w.bytes.size();
            }
            return static_cast<unsigned long long>(written);
        };
        if (rearrange() != length) {
            std::printf("%u-byte blocks: the writes did not cover the piece; no time reported\n", layout->block_bytes);
            return 1;
        }
        std::printf("%u-byte blocks, %zu streams, %.0f MiB:\n", layout->block_bytes, layout->streams.size(),
                    static_cast<double>(bytes) / 1048576.0);
        report("rearranged", bytes, measure(rearrange));
        report("memcpy of the same bytes", bytes, measure([&] {
            std::memcpy(copy_to.data(), file.data(), bytes);
            // The copy is read by nobody; this keeps the compiler from
            // removing it.
            asm volatile("" : : "r"(copy_to.data()) : "memory");
        }));
    }
    return 0;
}

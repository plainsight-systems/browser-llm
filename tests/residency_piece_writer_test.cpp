#include <doctest/doctest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/formats/device_layout.h"
#include "core/residency/piece_writer.h"
#include "core/residency/routes.h"

using namespace bllm;
using formats::DeviceLayout;
using residency::PieceWriter;
using residency::Route;
using residency::Write;
using residency::WriteError;

namespace {

// Test-only: a layout whose last stream is 3 bytes wide, so a run can end
// partway through a word and its tail, held back or padded, is exercised.
// No listed format has one; the writer's rules allow it.
constexpr formats::Stream kOddStreams[] = {{3, 4}, {7, 4}, {0, 3}};
constexpr DeviceLayout kOddLayout{gguf::TensorType::F32, 11, kOddStreams};
// Test-only: already in device order, as F32 is, but 3 bytes a block, so a
// run of whole blocks need not end on a word and cannot go straight out.
constexpr formats::Stream kOddIdentityStreams[] = {{0, 3}};
constexpr DeviceLayout kOddIdentity{gguf::TensorType::F32, 3, kOddIdentityStreams};

struct Piece {
    const DeviceLayout* layout;
    std::uint64_t blocks;
    std::uint64_t gap_before;   // file bytes outside every route, before it
};

struct Fixture {
    std::vector<std::byte> file;
    std::vector<Route> routes;
    std::uint64_t buffer_size = 0;
};

constexpr std::byte kUntouched{0xEE};

std::uint64_t round_up(std::uint64_t n, std::uint64_t to) { return (n + to - 1) / to * to; }

// A file of deterministic pseudo-random bytes holding `pieces` in order, each
// routed into buffer 0 at the next 256-byte boundary, as a plan places them.
Fixture make(std::span<const Piece> pieces, std::uint64_t trailing = 5) {
    Fixture f;
    std::uint32_t state = 0x9E3779B9u;
    const auto bytes = [&](std::uint64_t n) {
        for (std::uint64_t i = 0; i < n; ++i) {
            state = state * 1664525u + 1013904223u;
            f.file.push_back(static_cast<std::byte>(state >> 24));
        }
    };
    bytes(24);   // a header
    std::uint32_t tensor = 0;
    for (const Piece& p : pieces) {
        bytes(p.gap_before);
        const std::uint64_t stored = p.blocks * p.layout->block_bytes;
        f.routes.push_back({gguf::TensorId{tensor++}, f.file.size(), p.blocks, p.layout,
                            residency::BufferIndex{0}, f.buffer_size, round_up(stored, 4)});
        f.buffer_size = round_up(f.buffer_size + round_up(stored, 4), 256);
        bytes(stored);
    }
    bytes(trailing);
    return f;
}

// What the device should hold: each route's blocks gathered by hand into its
// streams, the last run zero-padded to the route's length, nothing else
// written. The untransformed reference CDSA.32 asks the conversion be
// tested against.
std::vector<std::byte> expected(const Fixture& f) {
    std::vector<std::byte> device(f.buffer_size, kUntouched);
    for (const Route& r : f.routes) {
        std::fill_n(device.begin() + static_cast<std::ptrdiff_t>(r.buffer_offset), r.length, std::byte{0});
        std::uint64_t at = r.buffer_offset;
        for (const formats::Stream s : r.layout->streams) {
            for (std::uint64_t b = 0; b < r.blocks; ++b) {
                for (std::uint64_t i = 0; i < s.width; ++i) {
                    device[at++] = f.file[r.file_offset + b * r.layout->block_bytes + s.offset + i];
                }
            }
        }
    }
    return device;
}

// Applies one write, checking it is word-aligned, a whole number of words,
// and inside one route's binding.
void apply(const Fixture& f, const Write& w, std::vector<std::byte>& device) {
    CHECK(w.offset % 4 == 0);
    CHECK(w.bytes.size() % 4 == 0);
    const bool inside = std::any_of(f.routes.begin(), f.routes.end(), [&](const Route& r) {
        return w.offset >= r.buffer_offset && w.offset + w.bytes.size() <= r.buffer_offset + r.length;
    });
    CHECK(inside);
    REQUIRE(w.offset + w.bytes.size() <= device.size());
    std::copy(w.bytes.begin(), w.bytes.end(), device.begin() + static_cast<std::ptrdiff_t>(w.offset));
}

// Streams the file through a writer in chunks of `chunk_size` and returns
// what the device would hold.
std::vector<std::byte> upload(const Fixture& f, std::size_t chunk_size) {
    PieceWriter writer(f.routes, chunk_size);
    std::vector<std::byte> device(f.buffer_size, kUntouched);
    for (std::size_t at = 0; at < f.file.size(); at += chunk_size) {
        const auto chunk = std::span(f.file).subspan(at, std::min(chunk_size, f.file.size() - at));
        std::vector<Write> writes;
        REQUIRE(writer.accept(at, chunk, writes) == WriteError::Ok);
        for (const Write& w : writes) apply(f, w, device);
    }
    CHECK(writer.finish(f.file.size()) == WriteError::Ok);
    return device;
}

bool points_into(std::span<const std::byte> inner, std::span<const std::byte> outer) {
    return std::less_equal<>{}(outer.data(), inner.data()) &&
           std::less_equal<>{}(inner.data() + inner.size(), outer.data() + outer.size());
}

const Piece kEveryLayout[] = {
    {&formats::kF32Layout, 7, 0},  {&formats::kQ4_0Layout, 5, 13}, {&formats::kQ4_1Layout, 3, 1},
    {&formats::kQ8_0Layout, 1, 0}, {&formats::kQ6_KLayout, 3, 30}, {&kOddLayout, 9, 2},
    {&formats::kQ4_0Layout, 1, 0}, {&kOddLayout, 1, 0},  {&kOddIdentity, 7, 0},
    {&kOddIdentity, 4, 3},
};

}  // namespace

TEST_CASE("the device holds every stored block's fields as its layout's streams, whatever the chunks") {
    const Fixture f = make(kEveryLayout);
    const auto want = expected(f);
    for (const std::size_t chunk : {std::size_t{1}, std::size_t{3}, std::size_t{4}, std::size_t{17},
                                    std::size_t{18}, std::size_t{210}, std::size_t{211}, std::size_t{4096},
                                    f.file.size()}) {
        CAPTURE(chunk);
        CHECK(upload(f, chunk) == want);
    }
}

TEST_CASE("the same chunks always give the same writes") {
    const Fixture f = make(kEveryLayout);
    const auto run = [&] {
        PieceWriter writer(f.routes, 64);
        std::vector<std::pair<std::uint64_t, std::vector<std::byte>>> seen;
        for (std::size_t at = 0; at < f.file.size(); at += 64) {
            std::vector<Write> writes;
            const auto chunk = std::span(f.file).subspan(at, std::min<std::size_t>(64, f.file.size() - at));
            REQUIRE(writer.accept(at, chunk, writes) == WriteError::Ok);
            for (const Write& w : writes) seen.push_back({w.offset, {w.bytes.begin(), w.bytes.end()}});
        }
        return seen;
    };
    CHECK(run() == run());
}

TEST_CASE("a piece within one chunk takes one write per stream") {
    for (const DeviceLayout* layout : {&formats::kQ4_0Layout, &formats::kQ6_KLayout, &kOddLayout}) {
        const Piece piece[] = {{layout, 6, 0}};
        const Fixture f = make(piece);
        PieceWriter writer(f.routes, f.file.size());
        std::vector<Write> writes;
        REQUIRE(writer.accept(0, f.file, writes) == WriteError::Ok);
        CHECK(writes.size() == layout->streams.size());
    }
}

TEST_CASE("a layout already in device order is written straight from the chunk") {
    const Piece piece[] = {{&formats::kF32Layout, 16, 0}};
    const Fixture f = make(piece);
    PieceWriter writer(f.routes, f.file.size());
    std::vector<Write> writes;
    REQUIRE(writer.accept(0, f.file, writes) == WriteError::Ok);
    REQUIRE(writes.size() == 1);
    CHECK(points_into(writes[0].bytes, f.file));
    CHECK(writes[0].bytes.size() == 64);
}

TEST_CASE("a chunk out of order, or too large, is refused; out is untouched and nothing more is accepted") {
    const Fixture f = make(kEveryLayout);
    const auto chunk = std::span(f.file).first(64);

    PieceWriter skipped(f.routes, 64);
    std::vector<Write> writes{Write{}};
    CHECK(skipped.accept(1, chunk, writes) == WriteError::OutOfOrder);
    CHECK(writes.size() == 1);
    CHECK(skipped.accept(0, chunk, writes) == WriteError::OutOfOrder);
    CHECK(skipped.finish(f.file.size()) == WriteError::OutOfOrder);

    PieceWriter small(f.routes, 63);
    CHECK(small.accept(0, chunk, writes) == WriteError::ChunkTooLarge);
    CHECK(writes.size() == 1);
    CHECK(small.accept(0, chunk.first(10), writes) == WriteError::ChunkTooLarge);
}

TEST_CASE("the file ending before every route is filled, or short of its size, is Unfinished") {
    const Fixture f = make(kEveryLayout);
    std::vector<Write> writes;

    PieceWriter early(f.routes, f.file.size());
    REQUIRE(early.accept(0, std::span(f.file).first(f.file.size() - 40), writes) == WriteError::Ok);
    CHECK(early.finish(f.file.size() - 40) == WriteError::Unfinished);

    // Every route filled, but the chunks stopped before the file's end.
    PieceWriter short_of_end(f.routes, f.file.size());
    REQUIRE(short_of_end.accept(0, std::span(f.file).first(f.file.size() - 1), writes) == WriteError::Ok);
    CHECK(short_of_end.finish(f.file.size()) == WriteError::Unfinished);
}

TEST_CASE("staging holds the case its bound is set by: a held block, then many padded one-block routes") {
    // Q6_K's 210-byte block cut by the first chunk's end, then 300 adjacent
    // one-block routes of the odd layout, each ending inside the full second
    // chunk with its last stream padded.
    std::vector<Piece> pieces{{&formats::kQ6_KLayout, 2, 0}};
    for (int i = 0; i < 300; ++i) pieces.push_back({&kOddLayout, 1, 0});
    const Fixture f = make(pieces, 0);
    const std::size_t first = 24 + 210 + 100;   // ends inside the second Q6_K block
    const std::size_t max_chunk = f.file.size() - first;
    PieceWriter writer(f.routes, max_chunk);
    std::vector<std::byte> device(f.buffer_size, kUntouched);

    std::vector<Write> writes;
    REQUIRE(writer.accept(0, std::span(f.file).first(first), writes) == WriteError::Ok);
    for (const Write& w : writes) apply(f, w, device);

    writes.clear();
    const auto chunk = std::span(f.file).subspan(first);
    REQUIRE(chunk.size() == max_chunk);
    REQUIRE(writer.accept(first, chunk, writes) == WriteError::Ok);
    std::uint64_t staged = 0;
    for (const Write& w : writes) {
        if (!points_into(w.bytes, chunk)) staged += w.bytes.size();
        apply(f, w, device);
    }
    CHECK(staged <= max_chunk + formats::kMaxBlockBytes + f.routes.size() * formats::kMaxStreams * 6);
    CHECK(staged > max_chunk);   // the padding and the held block's bytes are counted
    CHECK(writer.finish(f.file.size()) == WriteError::Ok);
    CHECK(device == expected(f));
}

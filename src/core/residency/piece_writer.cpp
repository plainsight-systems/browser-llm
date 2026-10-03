#include "core/residency/piece_writer.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "core/gguf/checked.h"

namespace bllm::residency {
namespace {

using formats::DeviceLayout;

constexpr std::uint64_t round_up4(std::uint64_t n) { return (n + 3) / 4 * 4; }

// Where stream `s` of a piece of `blocks` blocks starts, from the piece's
// start: after every earlier stream's fields. Each earlier stream is a
// multiple of 4 wide (device_layout.h), so every stream starts on a word.
std::uint64_t stream_start(const DeviceLayout& layout, std::size_t s, std::uint64_t blocks) {
    std::uint64_t before = 0;
    for (std::size_t i = 0; i < s; ++i) before += layout.streams[i].width;
    return before * blocks;
}

// Copies one field of every block in `run` into `out`, back to back, and
// returns the bytes copied.
template <std::size_t Width>
std::size_t gather_fixed(std::span<const std::byte> run, std::size_t block_bytes, std::size_t offset,
                         std::span<std::byte> out) {
    std::byte* to = out.data();
    for (std::size_t b = 0; b < run.size(); b += block_bytes, to += Width) {
        std::memcpy(to, run.data() + b + offset, Width);
    }
    return static_cast<std::size_t>(to - out.data());
}

// Optimization (practice): every field a listed layout has is one of these
// widths, so its copy has a size the compiler knows and inlines as a few
// loads and stores. Copied at a width known only at run time, each field
// was a call to memcpy: 21 million calls a stream for a Q4_0 model's
// weights. GDSA.6 prices this stage against memcpy of the same bytes, which
// a gather of small fields can reach about half of. make bench
// (bench/piece_writer_bench.cpp), 360 MiB, native release, Apple M3 Max:
//   Q4_0  83.3 ms (4.5 GB/s)  ->  11.9 ms (31.8 GB/s)
//   Q4_1 101.4 ms (3.7 GB/s)  ->  11.1 ms (34.2 GB/s)
//   Q8_0  49.6 ms (7.6 GB/s)  ->  10.7 ms (35.4 GB/s)
//   Q6_K  33.2 ms (11.4 GB/s) ->  13.7 ms (27.6 GB/s)
//   memcpy of the same bytes: 63 GB/s
std::size_t gather(std::span<const std::byte> run, std::size_t block_bytes, formats::Stream stream,
                   std::span<std::byte> out) {
    switch (stream.width) {
        case 2: return gather_fixed<2>(run, block_bytes, stream.offset, out);
        case 4: return gather_fixed<4>(run, block_bytes, stream.offset, out);
        case 16: return gather_fixed<16>(run, block_bytes, stream.offset, out);
        case 32: return gather_fixed<32>(run, block_bytes, stream.offset, out);
        case 64: return gather_fixed<64>(run, block_bytes, stream.offset, out);
        case 128: return gather_fixed<128>(run, block_bytes, stream.offset, out);
        default: break;   // a width no listed layout has: copied as it comes
    }
    std::size_t at = 0;
    for (std::size_t b = 0; b < run.size(); b += block_bytes, at += stream.width) {
        std::memcpy(out.data() + at, run.data() + b + stream.offset, stream.width);
    }
    return at;
}

// A layout of one stream that is the whole block: the device order is the
// file's, so a run of whole blocks is already its stream.
bool is_identity(const DeviceLayout& layout) {
    return layout.streams.size() == 1 && layout.streams[0].offset == 0 &&
           layout.streams[0].width == layout.block_bytes;
}

// Takes `n` bytes of staging past `staged`. The bound the staging is sized at
// is proved in piece_writer.h; exceeding it would mean the proof is wrong,
// which no caller can recover from, so it stops here rather than write past
// the area (E.26).
std::span<std::byte> take(std::span<std::byte> staging, std::size_t& staged, std::size_t n) {
    if (n > staging.size() - staged) std::abort();
    const auto area = staging.subspan(staged, n);
    staged += n;
    return area;
}

}  // namespace

PieceWriter::PieceWriter(std::span<const Route> routes, std::size_t max_chunk)
    : routes_(routes),
      max_chunk_(max_chunk),
      staging_(staging_bound(routes.size(), max_chunk)) {
    while (route_ < routes_.size() && routes_[route_].blocks == 0) ++route_;
}

WriteError PieceWriter::accept(std::uint64_t file_offset, std::span<const std::byte> chunk,
                               std::vector<Write>& out) {
    if (failed_ != WriteError::Ok) return failed_;
    if (chunk.size() > max_chunk_) return failed_ = WriteError::ChunkTooLarge;
    std::uint64_t end = 0;
    if (file_offset != next_offset_ || !gguf::checked_add(file_offset, chunk.size(), end)) {
        return failed_ = WriteError::OutOfOrder;
    }

    std::size_t staged = 0;
    std::uint64_t pos = file_offset;
    while (route_ < routes_.size() && pos < end) {
        const Route& route = routes_[route_];
        const std::uint64_t block_bytes = route.layout->block_bytes;
        // The route's next byte; routes are in file order and apart, so the
        // chunk never starts past it. Bytes before it are outside every route.
        const std::uint64_t next = route.file_offset + blocks_done_ * block_bytes + held_count_;
        if (pos < next) {
            pos = std::min(end, next);
            continue;
        }
        const std::uint64_t stop = std::min(end, route.file_offset + route.blocks * block_bytes);
        auto bytes = chunk.subspan(pos - file_offset, stop - pos);
        pos = stop;

        // A block cut off at the last chunk's end is completed first.
        std::span<const std::byte> held;
        if (held_count_ > 0) {
            const auto fill = std::min<std::size_t>(block_bytes - held_count_, bytes.size());
            std::copy_n(bytes.begin(), fill, held_.begin() + held_count_);
            held_count_ += fill;
            bytes = bytes.subspan(fill);
            if (held_count_ < block_bytes) continue;   // the chunk ended inside it
            held = std::span<const std::byte>(held_).first(block_bytes);
        }
        const auto whole = bytes.first(bytes.size() / block_bytes * block_bytes);
        if (!held.empty() || !whole.empty()) write_blocks(held, whole, staged, out);
        held_count_ = 0;

        // What is left is less than a block, cut off at the chunk's end.
        const auto rest = bytes.subspan(whole.size());
        std::copy(rest.begin(), rest.end(), held_.begin());
        held_count_ = rest.size();
    }
    next_offset_ = end;
    return WriteError::Ok;
}

void PieceWriter::write_blocks(std::span<const std::byte> held, std::span<const std::byte> blocks,
                               std::size_t& staged, std::vector<Write>& out) {
    const Route& route = routes_[route_];
    const DeviceLayout& layout = *route.layout;
    const std::uint64_t block_bytes = layout.block_bytes;
    const std::uint64_t count = held.size() / block_bytes + blocks.size() / block_bytes;
    const bool finished = blocks_done_ + count == route.blocks;

    for (std::size_t s = 0; s < layout.streams.size(); ++s) {
        const formats::Stream stream = layout.streams[s];
        Tail& tail = tails_[s];
        // The tail holds the run's bytes past its last whole word, so the
        // write starts on the word they begin.
        const std::uint64_t device = route.buffer_offset + stream_start(layout, s, route.blocks) +
                                     blocks_done_ * stream.width - tail.count;
        const std::uint64_t length = tail.count + count * stream.width;

        // Optimization (browser): a run already in device order, starting
        // and ending on a word, is written straight from the chunk; the
        // browser copies it once, and nothing is copied here first.
        if (is_identity(layout) && held.empty() && tail.count == 0 && length % 4 == 0) {
            out.push_back({route.buffer, device, blocks});
            continue;
        }

        const auto area = take(staging_, staged, round_up4(length));
        std::size_t at = std::copy_n(tail.bytes.begin(), tail.count, area.begin()) - area.begin();
        for (const auto run : {held, blocks}) {
            at += gather(run, block_bytes, stream, area.subspan(at));
        }

        std::uint64_t written = length;
        if (finished) {
            // The run that ends the piece is padded to the word; the piece's
            // bound length covers it (device_layout.h).
            std::fill(area.begin() + length, area.end(), std::byte{0});
            written = area.size();
            tail.count = 0;
        } else {
            written = length / 4 * 4;
            tail.count = static_cast<std::uint8_t>(length - written);
            std::copy_n(area.begin() + written, tail.count, tail.bytes.begin());
        }
        if (written > 0) out.push_back({route.buffer, device, area.first(written)});
    }

    blocks_done_ += count;
    if (finished) {
        blocks_done_ = 0;
        tails_ = {};
        do ++route_;
        while (route_ < routes_.size() && routes_[route_].blocks == 0);
    }
}

WriteError PieceWriter::finish(std::uint64_t file_size) {
    if (failed_ != WriteError::Ok) return failed_;
    if (route_ < routes_.size() || next_offset_ != file_size) return failed_ = WriteError::Unfinished;
    return WriteError::Ok;
}

}  // namespace bllm::residency

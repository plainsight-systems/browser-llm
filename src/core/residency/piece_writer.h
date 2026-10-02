#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/formats/device_layout.h"
#include "core/residency/routes.h"

namespace bllm::residency {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// Turns the file, as it arrives in chunks, into the writes that fill the
// planned buffers. A deterministic state machine with no I/O: it holds no GPU
// object and says what to write, not how; upload issues each write as one
// wgpuQueueWriteBuffer (upload.h), and every rule here is tested without a
// device. The same chunks always give the same writes, so the diagnostic
// check runs it again to know what the device should hold (upload_check.h).
//
//   - Chunks arrive in order, of any size; a block, a row or a piece may
//     span chunks. The bytes of a block cut off at a chunk's end are held
//     until the next chunk completes it — never more than one block (210
//     bytes, Q6_K's), so what is held does not grow with the file.
//   - A chunk's whole blocks of a piece are laid out as the piece's device
//     layout says: each stream's run of fields, contiguous, in one write. A
//     piece in a 16 MiB chunk takes one write per stream, not one per block.
//     Optimization (browser): every write crosses into the browser and is
//     validated there (WASM.2); writes scale with streams, not with blocks.
//   - A layout of one stream (F32) needs no rearranging: its write points
//     into the chunk itself, and nothing is copied.
//   - Every write starts on a 4-byte boundary and is a multiple of 4 bytes
//     long, as writeBuffer requires. A stream's run that ends partway through
//     a word holds back those bytes for the next chunk; the run that ends the
//     piece is padded with zeros to the word, which the piece's bound length
//     covers.
//   - Rearranged bytes are written from one staging area, allocated once and
//     reused: the heap holds a chunk and its staging, whatever the file
//     (WASM.1, WASM.9, MEM.9). Its size is proved from the routes, not
//     guessed: what one chunk stages is at most the chunk itself, plus one
//     block held over from the chunk before, plus, for each route the chunk
//     reaches, up to 3 bytes held back and 3 bytes of padding in each of its
//     streams. Every route could end inside one chunk — a file of adjacent
//     one-block tensors does — so the bound counts all of them:
//       max_chunk + kMaxBlockBytes + routes × kMaxStreams × 6,
//     8,418 bytes beyond the chunk for Gemma 3's 342 routes. A native test
//     drives the case the bound is set by — a full chunk that completes a
//     held block, then many adjacent one-block routes, each padded — and
//     checks what it staged against the bound.
//   - Bytes outside every route — the header, the padding between tensors —
//     are skipped. A chunk that does not start where the last one ended, or
//     the file ending with a route unfilled, is a named failure.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     C.2    Use class if the class has an invariant — the members below, and
//            the invariant each keeps.
//     E.4    Design your error-handling strategy around invariants — after a
//            failure the writer accepts nothing, so no write follows a broken
//            invariant.
//     E.27   Use error codes systematically — WriteError.
//     R.14   Prefer span to pointer and count — chunks in, writes out.
//   C++ performance guidelines
//     WASM.1 Size linear memory to the real high-water mark — one chunk and
//            its staging, allocated once.
//     WASM.2 Batch work across the JS boundary — a write per stream, not per
//            block.
//     WASM.9 Stream in bounded chunks — what is held is at most one block.
//     MEM.9  Allocate at init, not in steady state — the staging area.
//     CDSA.32 Transform static data once into the layout its consumer reads —
//            this is that transform. Its caveat, that a load-time transform
//            doubles peak memory while source and result coexist, is met by
//            transforming a chunk at a time; and its conversion is tested
//            against the untransformed reference: every stream's bytes,
//            gathered back, are the stored blocks' fields in block order.

// One write: `bytes` at `offset` in `buffer`. The span borrows, never owns:
// it points into the chunk accept was given, valid for as long as the caller
// keeps that chunk, or into the writer's staging, valid until the next call
// to accept. A caller that holds writes past the call — the diagnostic check
// does, until its comparison completes — keeps the chunk, and makes no other
// call to accept, until then (I.11: ownership stays with the caller).
struct Write {
    BufferIndex buffer;
    std::uint64_t offset;
    std::span<const std::byte> bytes;
};

enum class WriteError {
    Ok,
    OutOfOrder,         // a chunk did not start where the last one ended
    ChunkTooLarge,      // larger than the staging the writer was given
    Unfinished,         // the file ended before every route was filled
};

class PieceWriter {
public:
    // `routes` in file order, as plan_routes gives them; `max_chunk` the
    // largest chunk accept will be given. Precondition: `routes` outlives
    // the writer.
    PieceWriter(std::span<const Route> routes, std::size_t max_chunk);

    // The writes the chunk at `file_offset` completes, appended to `out`. On
    // failure `out` is left as it was and the writer accepts nothing more.
    [[nodiscard]] WriteError accept(std::uint64_t file_offset, std::span<const std::byte> chunk,
                                    std::vector<Write>& out);

    // Called once the file has ended at `file_size`: Unfinished unless the
    // chunks reached it and every route was filled. Unfinished is a failure
    // like the others: from then on accept refuses every chunk.
    [[nodiscard]] WriteError finish(std::uint64_t file_size);

private:
    // Bytes of one stream's run held back because it ended partway through a
    // word; they lead that stream's next write.
    struct Tail {
        std::array<std::byte, 3> bytes{};
        std::uint8_t count = 0;
    };

    // Lays out route_'s next blocks — `held`, one block or none, then
    // `blocks` — as its streams: a write per stream, appended to `out`, from
    // the chunk itself or from staging_ past `staged`, which it advances.
    void write_blocks(std::span<const std::byte> held, std::span<const std::byte> blocks,
                      std::size_t& staged, std::vector<Write>& out);

    std::span<const Route> routes_;
    std::size_t max_chunk_;
    std::size_t route_ = 0;            // the first route not yet filled
    std::uint64_t blocks_done_ = 0;    // of route_, the blocks already written
    std::uint64_t next_offset_ = 0;    // where the next chunk must start
    // A block of route_ cut off at the last chunk's end: fewer than its
    // layout's block_bytes, so never more than kMaxBlockBytes - 1.
    std::array<std::byte, formats::kMaxBlockBytes> held_{};
    std::size_t held_count_ = 0;
    std::array<Tail, formats::kMaxStreams> tails_{};   // one for each of route_'s streams
    // Where rearranged bytes are written from: allocated once, at
    // construction, at the bound above, and never grown (MEM.9).
    std::vector<std::byte> staging_;
    // Set by the first failure; from then on accept refuses every chunk.
    WriteError failed_ = WriteError::Ok;
};

}  // namespace bllm::residency

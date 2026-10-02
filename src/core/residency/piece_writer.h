#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/residency/routes.h"

namespace bllm::residency {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// Turns the file, as it arrives in chunks, into the writes that fill the
// planned buffers. Pure: it holds no GPU object and says what to write, not
// how; upload issues each write as one wgpuQueueWriteBuffer (upload.h), and
// every rule here is tested without a device.
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
//     piece is padded with zeros to the word, which the piece's length allows.
//   - Rearranged bytes are written from one staging area, allocated once at
//     the size of the largest chunk and reused: the heap holds a chunk and its
//     staging, whatever the file (WASM.1, WASM.9, MEM.9).
//   - Bytes outside every route — the header, the padding between tensors —
//     are skipped. A chunk that does not start where the last one ended, or
//     the file ending with a route unfilled, is a named failure.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     F.8    Prefer pure functions — it decides writes and issues none.
//     C.2    Use class if the class has an invariant — the bytes held never
//            exceed one block, and the next route is the first unfilled.
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

// One write: `bytes` at `offset` in `buffer`. The span points into the chunk
// or into the writer's staging, and is valid until the next call to accept.
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

    // Called once the file has ended at `file_size`: Unfinished unless every
    // route was filled.
    [[nodiscard]] WriteError finish(std::uint64_t file_size) const;
};

}  // namespace bllm::residency

#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/residency/piece_writer.h"

namespace bllm::residency {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// The Upload stage's own test, in a diagnostic build only: every byte of every
// weight is read back from the device and checked against what was written.
// A sampled check passes while another chunk sits at the wrong offset; WASM.9
// asks for every byte or no claim of integrity.
//
//   - As upload issues writes, each buffer's expected contents are folded
//     into a digest: every nonzero 32-bit word contributes a mix of its
//     offset and its value, and the contributions are summed. A sum does not
//     depend on the order the writes came in — streams fill a piece out of
//     address order — and a zero word contributes nothing, so padding the
//     writes never touched, which WebGPU zeroed, agrees on both sides.
//   - After upload, each weight buffer is copied in 16 MiB slices into one
//     mappable staging buffer, mapped, and folded the same way; the slices
//     bound the readback's memory, as the chunks bound the upload's. It is the
//     one round trip of the whole model's bytes, and only this build makes it
//     (GPU.1).
//   - A digest is not a byte comparison: a mismatch is certain, a match
//     wrong only by a collision in a 64-bit mix (splitmix64's finalizer),
//     which no layout bug produces on purpose.
//
// Compiled only when BLLM_DIAGNOSTICS_ENABLED; the shipped module holds no
// trace of it, and a scan of the artifact proves that (TLM.8).
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     P.6    What cannot be checked at compile time should be checkable at run
//            time — where the bytes landed is checked on the device itself.
//   C++ performance guidelines
//     WASM.9 Verify every byte, or claim no integrity — every word of every
//            weight buffer is folded, none sampled.
//     GPU.1  Budget every round trip — the whole model is read back once, in
//            this build only.
//     TLM.8  Validate clean builds by artifact scan — the shipped module is
//            checked to hold none of this.

class BufferDigests {
public:
    explicit BufferDigests(std::size_t buffers);

    // Folds the writes into their buffers' digests.
    void add(std::span<const Write> writes);

    // Folds `bytes`, read back from `buffer` starting at `offset`, into a
    // second digest for it.
    void add_readback(BufferIndex buffer, std::uint64_t offset, std::span<const std::byte> bytes);

    // The buffers whose readback does not match what was written.
    [[nodiscard]] std::vector<BufferIndex> mismatches() const;
};

}  // namespace bllm::residency

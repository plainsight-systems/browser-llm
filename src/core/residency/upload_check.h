#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <webgpu/webgpu.h>

#include "core/gpu/wgpu_handles.h"
#include "core/residency/piece_writer.h"
#include "core/residency/upload.h"

namespace bllm::residency {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// The Upload stage's own test, in a diagnostic build only: every byte upload
// wrote is read back from the device and compared, byte for byte, with what
// it should be. A sampled check passes while another chunk sits at the wrong
// offset; WASM.9 asks for every byte or no claim of integrity.
//
//   - What each range should hold is regenerated, not remembered: once upload
//     has finished, the page streams the cached file a second time, and a
//     fresh PieceWriter over the routes the Upload itself used — taken from
//     it, never passed in beside it, so the check cannot be pointed at other
//     routes and pass having compared nothing — turns it into the same writes
//     upload issued; it is deterministic (piece_writer.h). Nothing the size of
//     the model is kept to compare against.
//   - Every comparison rests on a completed mapping, which a lost device
//     refuses (upload.h), so a check on a lost device fails and never ends
//     with zero mismatches having compared nothing. A chunk's mapping maps as
//     the witness's does: CallbackCancelled is Cancelled; Error, and Success
//     with no range, are Internal; Aborted is DeviceLost where Upload's
//     device_status() already says so, and otherwise MapFailed. Its callbacks keep
//     their state alive on their own, as Upload's do: its destructor marks
//     them cancelled, so destroying a check with a comparison pending reports
//     Cancelled instead of touching freed memory.
//   - For each chunk, every write's range is copied into one mappable staging
//     buffer, back to back, in one command buffer; the staging buffer is
//     mapped once, and each range is compared with the write's bytes. The
//     chunk is acknowledged only after that comparison, so the chunk and the
//     writer's staging the writes point into are still intact when compared.
//     The readback is bounded by a chunk, as the upload was.
//   - The result names every mismatch: the buffer, the offset of its first
//     differing byte, and the tensor the route belongs to, by the name the
//     Upload kept. Zero mismatches
//     means every byte written is on the device where the plan put it.
//     Padding upload never wrote is not compared: WebGPU zeroes new buffers,
//     and nothing reads past a piece's file bytes but its unpack.
//
// Compiled only when BLLM_DIAGNOSTICS_ENABLED; the shipped module holds no
// trace of it, and a scan of the artifact proves that (TLM.8).
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     P.6    What cannot be checked at compile time should be checkable at run
//            time — where the bytes landed is checked on the device itself.
//     R.1    Manage resources automatically using RAII — the staging buffer
//            and the device reference are RAII handles.
//     E.27   Use error codes systematically — CheckError, and mismatches as
//            data, never a log line.
//   C++ performance guidelines
//     WASM.9 Stream assets in bounded chunks; its Caveats: "A sampled
//            verification proves very little... Verify every byte, in a
//            diagnostic pass, or do not claim integrity." — every byte written
//            is compared, none sampled, a chunk at a time.
//     GPU.1  Budget every round trip — the whole model is read back once, in
//            this build only.
//     TLM.8  Validate clean builds by artifact scan — the shipped module is
//            checked to hold none of this.

enum class CheckError {
    Ok,
    Cancelled,          // the check was destroyed before the comparison ended
    DeviceLost,         // a mapping was aborted, and the device reports itself lost
    MapFailed,          // a mapping was aborted, and the device has not said why
    Internal,           // a mapping this check should not have asked for, or one with no range
    OutOfOrder,         // a chunk did not start where the last one ended
    ChunkTooLarge,
    Unfinished,         // the file ended before every route was compared
};

struct Mismatch {
    BufferIndex buffer;
    std::uint64_t offset;   // the first differing byte, within the buffer
    std::string tensor;
};

using CheckCallback = void (*)(CheckError error, void* userdata);

class UploadCheck {
public:
    UploadCheck(const UploadCheck&) = delete;
    UploadCheck& operator=(const UploadCheck&) = delete;

    // Checks the buffers `upload` filled, with the routes it used, naming
    // tensors by the names it kept. Takes its own reference to upload's
    // device. Preconditions: `upload` has finished, and outlives the check —
    // it owns the buffers being checked, so it must anyway. `max_chunk` as for
    // Upload.
    UploadCheck(const Upload& upload, std::size_t max_chunk);

    // Marks pending callbacks cancelled, then releases the staging buffer.
    ~UploadCheck() noexcept;

    // Compares the ranges the chunk at `file_offset` covers; `accepted` once
    // they are compared and the page may send the next.
    void check(std::uint64_t file_offset, std::span<const std::byte> chunk, CheckCallback accepted,
               void* userdata);

    // Called after the last chunk: Unfinished unless every route was compared.
    [[nodiscard]] CheckError finish(std::uint64_t file_size) const;

    // Every mismatch found so far, in file order.
    [[nodiscard]] std::span<const Mismatch> mismatches() const noexcept { return mismatches_; }

private:
    gpu::DeviceHandle device_;
    const Upload& upload_;   // finished; outlives the check
    PieceWriter writer_;      // regenerates upload's writes
    gpu::Buffer staging_;     // MAP_READ | COPY_DST, sized to the largest chunk's writes
    std::vector<Write> writes_;
    std::vector<Mismatch> mismatches_;
    std::shared_ptr<struct CheckState> pending_;   // shared with callbacks in flight
    CheckError failed_ = CheckError::Ok;
};

}  // namespace bllm::residency

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <webgpu/webgpu.h>

#include "core/gpu/wgpu_handles.h"
#include "core/residency/piece_writer.h"
#include "core/residency/plan.h"
#include "core/residency/routes.h"

namespace bllm::residency {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// Carries out a residency plan on a device: creates every planned buffer,
// fills the weights from the file as it streams in, and owns the buffers for
// the life of the loaded model. The KV cache and the kernels refer to them by
// their index in the plan. What goes where is decided without a device, by
// routes.h and piece_writer.h; this is the part that touches the GPU.
//
//   - Buffers are created up front, every one the plan names except those of
//     a confirmed duplicate: a few large buffers, suballocated by the plan
//     (GPU.9). Weights are STORAGE | COPY_DST; the cache STORAGE; working
//     buffers STORAGE | COPY_SRC, for the sampled token's readback. A
//     diagnostic build adds COPY_SRC to the weights, to read them back
//     (upload_check.h). WebGPU zeroes a new buffer, so padding is never
//     written.
//   - Creation is checked: it runs inside out-of-memory and validation error
//     scopes, and ready is reported only once both scopes are popped clean.
//     A buffer the device could not give is a named failure before any byte
//     is written, never a write into an invalid buffer (E.27).
//   - Each write is one wgpuQueueWriteBuffer, from the chunk or the staging
//     area in the wasm heap. writeBuffer copies the bytes before it returns,
//     so the heap's chunk and staging are free for the next chunk at once.
//   - At most two chunks are in flight. A chunk is acknowledged when the
//     queue has finished the chunk before it (wgpuQueueOnSubmittedWorkDone),
//     and the page sends the next only on acknowledgement, so the browser
//     stages at most two chunks' bytes for the GPU while the page reads the
//     next from the cache.
//     Optimization (practice): reading chunk n + 1 overlaps copying chunk n,
//     with in-flight memory bounded, as GPU.7 describes; waiting on every
//     chunk would leave the GPU's copy engine idle between them.
//   - Failure is a value (E.27): out of memory, the device lost, a chunk out
//     of order, or the file ending short. After a failure, later calls report
//     it and write nothing; the buffers are released with the Upload (R.1).
//   - Completion is reported through callbacks and never waited for: the
//     build does not use ASYNCIFY, and the worker must stay responsive.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     R.1    Manage resources automatically using RAII — every buffer is a
//            gpu::Buffer, released with the Upload.
//     I.11, R.20  Never transfer ownership by a raw pointer; use unique_ptr
//            to represent ownership — begin hands the Upload to its callback.
//     E.27   Use error codes systematically — UploadError, and every call
//            after a failure reports it.
//   C++ performance guidelines
//     GPU.9  Suballocate GPU memory from large heaps — a few large buffers,
//            as the plan packs them.
//     GPU.7  Pipeline CPU and GPU work with queues, fences and multi-buffered
//            resources — two chunks in flight, acknowledged on queue
//            completion, so the cache read of one overlaps the GPU copy of
//            the last.
//     GPU.1  Keep data on the device; budget every round trip — the shipped
//            path reads nothing back.

enum class UploadError {
    Ok,
    OutOfMemory,        // a planned buffer could not be created
    DeviceLost,
    OutOfOrder,         // a chunk did not start where the last one ended
    ChunkTooLarge,
    Unfinished,         // the file ended before every weight was filled
    OutOfRange,         // routes.h refused the plan; see the route's subject
};

// Invoked exactly once per call that takes it, from the browser's event loop.
using UploadCallback = void (*)(UploadError error, void* userdata);

class Upload;

// Delivers the upload, or null and why, exactly once — the shape of
// gpu::Device::request. `subject` names the tensor or buffer at fault; it is
// valid only during the call.
using ReadyCallback = void (*)(std::unique_ptr<Upload> upload, UploadError error, const char* subject,
                               void* userdata);

class Upload {
public:
    Upload(const Upload&) = delete;
    Upload& operator=(const Upload&) = delete;

    // Plans the routes, creates the buffers, and calls `ready` once the
    // device has confirmed it holds them. Ownership leaves by unique_ptr, so
    // nothing outlives the call by accident (I.11, R.20). `max_chunk` is the
    // largest chunk write will be given.
    static void begin(WGPUDevice device, const gguf::TensorIndex& index, const ResidencyPlan& plan,
                      std::uint64_t file_size, std::span<const gguf::TensorId> confirmed_duplicates,
                      std::size_t max_chunk, ReadyCallback ready, void* userdata);

    // Queues the writes the chunk at `file_offset` completes. `accepted` is
    // called when the page may send the next chunk: at once for the first,
    // and after the queue finishes the previous one for the rest.
    void write(std::uint64_t file_offset, std::span<const std::byte> chunk, UploadCallback accepted,
               void* userdata);

    // Called after the last chunk: `done` once every queued write has
    // completed, or with the error that stopped them.
    void finish(UploadCallback done, void* userdata);

    // The plan as carried out: confirmed duplicates read what they copy.
    [[nodiscard]] const ResidencyPlan& plan() const noexcept { return plan_; }

    // The created buffers, by their index in the plan; null where a buffer
    // was left uncreated.
    [[nodiscard]] WGPUBuffer buffer(BufferIndex index) const noexcept;

private:
    Upload(WGPUDevice device, ResidencyPlan plan, std::vector<Route> routes, std::size_t max_chunk);

    WGPUDevice device_;
    ResidencyPlan plan_;
    std::vector<Route> routes_;
    PieceWriter writer_;
    std::vector<gpu::Buffer> buffers_;
    std::vector<Write> writes_;   // reused across chunks
    UploadError failed_ = UploadError::Ok;
};

}  // namespace bllm::residency

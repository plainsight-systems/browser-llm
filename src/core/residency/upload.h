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
//     stages at most two chunks' bytes for the GPU whatever happens.
//     Optimization (practice): this is built so that the page's read of chunk
//     n + 1 can overlap the GPU's copy of chunk n, as GPU.7 describes;
//     waiting on every chunk would leave the copy engine idle between them.
//     Whether the overlap happens is the browser's to decide, so it is
//     claimed only for a target whose timeline shows it (measured below).
//   - Failure is a value (E.27), one for each cause, mapped exactly: the
//     out-of-memory scope's error is OutOfMemory, the validation scope's
//     Validation, an internal error Internal; a scope that cannot be popped,
//     and any callback whose status says the device is gone, DeviceLost; a
//     chunk out of order, too large, or the file ending short, their own
//     values; routes' refusals, theirs. After a failure, later calls report it
//     and write nothing; the buffers are released with the Upload (R.1).
//   - The state a callback needs lives apart from the Upload, shared between
//     the Upload and each callback still in flight (R.20, R.21: the one
//     shared ownership here, because either may end first). Destroying an
//     Upload with work queued marks that state cancelled; each pending
//     callback still fires exactly once, with Cancelled, and touches nothing
//     freed. The state holds its own device reference, so the device outlives
//     every callback too.
//   - Completion is reported through callbacks and never waited for: the
//     build does not use ASYNCIFY, and the worker must stay responsive.
//   - The Upload holds its own counted reference to the device
//     (wgpuDeviceAddRef, released by its RAII handle), so the device outlives
//     the Upload, whoever else lets it go.
//
// Measured, and reported with the target it ran on — desktop-chromium-floor
// (Chrome or Edge stable, an integrated GPU, WebGPU's default limits) and
// desktop-chromium-dev (the development machine), the matrix BLLM-002 set:
//   - Load throughput and the wasm heap's high-water mark, with chunks of 4,
//     16 and 64 MiB, in the release build. 16 MiB is the starting choice:
//     the size the page already reads the cache in.
//   - Whether reading overlaps copying, in the diagnostic build, as a
//     timeline: when the page has chunk n + 1 read, against when the queue
//     acknowledges chunk n. The overlap is real only where the read finishes
//     first (GPU.7); otherwise the two-chunk pipeline is not claimed.
//   - Release and diagnostic figures are reported apart, never mixed
//     (research/2026-08-31-measurement-build-configurations.md).
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     R.1    Manage resources automatically using RAII — every buffer is a
//            gpu::Buffer, and the device reference a gpu::DeviceHandle,
//            released with the Upload.
//     R.3    A raw pointer is non-owning — so the device is not held by one:
//            a raw WGPUDevice would leave its lifetime to someone else.
//     I.11, R.20  Never transfer ownership by a raw pointer; use unique_ptr
//            to represent ownership — begin hands the Upload to its callback.
//     R.21   Prefer unique_ptr over shared_ptr unless you need to share
//            ownership — the callback state is the one shared owner, since
//            the Upload or a callback in flight may end first.
//     E.27   Use error codes systematically — UploadError, one value per
//            failure, and every call after a failure reports it.
//   C++ performance guidelines
//     GPU.9  Suballocate GPU memory from large heaps — a few large buffers,
//            as the plan packs them.
//     GPU.7  Pipeline CPU and GPU work with queues, fences and multi-buffered
//            resources — two chunks in flight, acknowledged on queue
//            completion; claimed only once a timeline shows the overlap.
//     GPU.1  Keep data on the device; budget every round trip — the shipped
//            path reads nothing back.

// One value for each way upload can fail, routes' refusals among them, so a
// caller can tell a format this build lacks from a bad file (E.27).
enum class UploadError {
    Ok,
    Cancelled,          // the Upload was destroyed with this work pending
    UnsupportedFormat,  // routes.h: a tensor's format is not listed
    OutOfRange,         // routes.h: a piece reads past the file or its buffer
    NotACandidate,      // routes.h: a confirmed duplicate the plan never marked
    OutOfMemory,        // the out-of-memory scope caught an error
    Validation,         // the validation scope caught an error: a defect here, not in the file
    Internal,           // the device reported an internal error
    DeviceLost,         // the device was lost, or a scope could not be popped
    OutOfOrder,         // a chunk did not start where the last one ended
    ChunkTooLarge,
    Unfinished,         // the file ended before every weight was filled
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
    // `find_format` is capability::find_format in the harness (routes.h).
    static void begin(WGPUDevice device, const gguf::TensorIndex& index, const ResidencyPlan& plan,
                      std::uint64_t file_size, FindFormat find_format,
                      std::span<const gguf::TensorId> confirmed_duplicates, std::size_t max_chunk,
                      ReadyCallback ready, void* userdata);

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

    // The routes the upload carried out, in file order; upload_check.h
    // checks exactly these.
    [[nodiscard]] std::span<const Route> routes() const noexcept { return routes_; }

    [[nodiscard]] WGPUDevice device() const noexcept { return device_.get(); }

private:
    Upload(gpu::DeviceHandle device, ResidencyPlan plan, std::vector<Route> routes, std::size_t max_chunk);

    gpu::DeviceHandle device_;   // a reference of its own, taken in begin
    ResidencyPlan plan_;
    std::vector<Route> routes_;
    PieceWriter writer_;
    std::vector<gpu::Buffer> buffers_;
    std::vector<Write> writes_;   // reused across chunks
    std::shared_ptr<struct UploadState> pending_;   // shared with callbacks in flight
    UploadError failed_ = UploadError::Ok;
};

}  // namespace bllm::residency

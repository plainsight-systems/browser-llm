#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <webgpu/webgpu.h>

#include "core/gpu/device.h"
#include "core/gpu/wgpu_handles.h"
#include "core/residency/mapping.h"
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
//   - Creation is checked: it runs inside out-of-memory, validation and
//     internal error scopes, and ready reports only what those can show: a
//     buffer the device refused is a named failure before any byte is
//     written (E.27). Ready is not proof the device is alive — a lost device
//     pops its scopes clean — and nothing claims it is; finish is.
//   - Every write runs inside validation and internal error scopes, pushed
//     before the first write and popped by finish before the witness: a
//     write the device rejects never reaches the queue, so a witness written
//     after it could still map, and only a scope shows the rejection.
//   - Success is shown, never assumed. A lost device still reports queued
//     work as done and error scopes as clean, and WebGPU does not order its
//     lost callback before them, so neither a clean status nor the absence
//     of a loss report is evidence. A completed mapping is: a lost device
//     refuses one (tests/gpu/device_test.cpp holds that on every adapter CI
//     runs). So once the write phase's scopes pop clean, finish writes a
//     witness, four bytes unique to this upload into a small buffer of its
//     own, maps it, and reports Ok only if the mapping completes and reads
//     those bytes back: the queue runs in order, and no write was rejected,
//     so every write ran, on a live device. It costs one 4-byte round trip
//     per load, about half a millisecond (GPU.1).
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
//   - Failure is a value (E.27), one for each cause, mapped exactly from
//     webgpu.h:
//       - the witness, by witness_result below: its mapping read by
//         mapping_result (mapping.h), then its bytes. A mapping that is Ok
//         with the expected four bytes is Ok, the only Ok finish gives; Ok
//         with other bytes is WitnessMismatch; Cancelled, Internal and
//         DeviceLost are themselves; Unexplained is Unconfirmed.
//       - a popped scope's error, around creation or around the writes:
//         its type, as below.
//       - a popped scope's error type: OutOfMemory is OutOfMemory, Validation
//         is Validation, Internal and Unknown are Internal.
//       - a scope pop that fails: CallbackCancelled (the instance went away)
//         is Cancelled; Error (no scope to pop, a defect here) is Internal.
//       - queued work done: CallbackCancelled is Cancelled; Error (a queue
//         error) is Internal.
//       - a chunk out of order, too large, or the file ending short, and
//         routes' refusals: their own values.
//     After a failure, later calls report it and write nothing; the buffers
//     are released with the Upload (R.1).
//   - The state a callback needs lives apart from the Upload, shared between
//     the Upload and each callback still in flight (R.20, R.21: the one
//     shared ownership here, because either may end first). The Upload's
//     destructor marks that state cancelled before any member is released;
//     each pending callback still runs exactly once, reports Cancelled, and
//     touches only that state, which holds its own device reference and the
//     device's status. So a caller's userdata must stay valid until its
//     callback has run — after the Upload is destroyed, if work was pending.
//
// Tested: mapping_result and witness_result branch by branch, without a
// device; and on Dawn (tests/gpu), finish on a live device is Ok; a write
// the device rejects makes finish Validation, never Ok; finish after the
// device is destroyed is DeviceLost or Unconfirmed, never Ok; and destroying
// the Upload with the witness's mapping pending reports Cancelled, once; and
// releasing the caller's device and instance with work pending changes
// nothing: the work completes once, never CallbackCancelled.
//   - Completion is reported through callbacks and never waited for: the
//     build does not use ASYNCIFY, and the worker must stay responsive
//     (WASM.3).
//   - The Upload holds its own counted references to the device and to its
//     instance, taken with gpu::retain (wgpu_handles.h) and released by their
//     RAII handles, and so does the state its callbacks share: WebGPU's callbacks are the instance's, and dropping
//     an instance's last reference cancels them. So the Upload, and every
//     callback it has queued, outlive whoever else lets the device or the
//     instance go.
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
//   - Every figure carries its conditions: the clock's resolution observed,
//     whether the page was cross-origin isolated, that DevTools was closed,
//     and whether the load was the page's first (WASM.11).
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
//     C.21   If you define or =delete any copy, move, or destructor
//            function, define or =delete them all — its destructor cancels
//            pending callbacks, so copy and move are each deleted, not left
//            implicit; begin hands it out by unique_ptr.
//   C++ performance guidelines
//     COPY.4 Never declare only a destructor — the same four deletions.
//     GPU.9  Suballocate GPU memory from large heaps — a few large buffers,
//            as the plan packs them.
//     GPU.7  Pipeline CPU and GPU work with queues, fences and multi-buffered
//            resources — two chunks in flight, acknowledged on queue
//            completion; claimed only once a timeline shows the overlap.
//     WASM.3 Do not buy Asyncify to keep a blocking loop — completion
//            arrives by callback, and the worker returns to its event loop.
//     WASM.14 State the target matrix and budget for the weakest device —
//            desktop-chromium-floor, at WebGPU's default limits, is the
//            row every figure is reported for.
//     WASM.11 State the measurement conditions or the browser number means
//            nothing — the conditions above travel with each figure.
//     TLM.6  Diagnostic mode is not benchmark mode — the overlap timeline
//            comes from the diagnostic build and is never quoted as
//            throughput.
//     GPU.1  Keep data on the device; budget every round trip — the shipped
//            path reads back four bytes, once per load, as its proof of
//            success: a serialized round trip measured 0.5 ms median in
//            Chrome on Apple silicon (research/2026-08-31-gpu-readback-round-
//            trip.md); the witness path itself is measured on the target
//            matrix.

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
    Internal,           // an internal error, or a scope or queue error (see the mapping above)
    DeviceLost,         // the witness was refused, and the device reports itself lost
    Unconfirmed,        // the witness was refused, and the device has not said why
    WitnessMismatch,    // the witness mapped, but held other bytes than were written
    OutOfOrder,         // a chunk did not start where the last one ended
    ChunkTooLarge,
    Unfinished,         // the file ended before every weight was filled
};

// What finish reports for its witness: the mapping's outcome, by
// mapping_result, then the bytes it gave back, compared with those written.
// Deterministic, so every branch is tested without a device. `mapped` is
// consulted only when the mapping is Ok.
[[nodiscard]] UploadError witness_result(Mapped mapping, std::span<const std::byte> mapped,
                                         std::span<const std::byte, 4> expected) noexcept;

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
    Upload(Upload&&) = delete;
    Upload& operator=(Upload&&) = delete;

    // Plans the routes, creates the buffers, and calls `ready` once the
    // device has confirmed it holds them. Ownership leaves by unique_ptr, so
    // nothing outlives the call by accident (I.11, R.20). `max_chunk` is the
    // largest chunk write will be given.
    // `find_format` is capability::find_format in the harness (routes.h).
    // Takes its own reference to the device and shares its status.
    static void begin(const gpu::Device& device, const gguf::TensorIndex& index, const ResidencyPlan& plan,
                      std::uint64_t file_size, FindFormat find_format,
                      std::span<const gguf::TensorId> confirmed_duplicates, std::size_t max_chunk,
                      ReadyCallback ready, void* userdata);

    // Marks pending callbacks cancelled, then releases the buffers.
    ~Upload() noexcept;

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

    // The instance the device came from. WebGPU has no way to find it from the
    // device, so a check of this upload takes its own reference from here.
    [[nodiscard]] WGPUInstance instance() const noexcept { return instance_.get(); }

    // The device's status, to name the cause of a failure; never proof of
    // success (see above). Shared, as gpu::Device::status() is.
    [[nodiscard]] std::shared_ptr<const gpu::DeviceStatus> device_status() const noexcept {
        return device_status_;
    }

    // A routed tensor's name, kept from the index the routes came from, so a
    // report about a route needs no index beside the Upload.
    [[nodiscard]] std::string_view tensor_name(gguf::TensorId tensor) const noexcept;

private:
    Upload(gpu::DeviceHandle device, ResidencyPlan plan, std::vector<Route> routes, std::size_t max_chunk);

    gpu::Instance instance_;     // a reference of its own: its callbacks are the instance's
    gpu::DeviceHandle device_;   // a reference of its own, taken in begin
    gpu::Buffer witness_;        // MAP_READ | COPY_DST, 4 bytes, written last and read back
    std::shared_ptr<const gpu::DeviceStatus> device_status_;
    std::vector<std::string> tensor_names_;   // by TensorId, for the routed tensors
    ResidencyPlan plan_;
    std::vector<Route> routes_;
    PieceWriter writer_;
    std::vector<gpu::Buffer> buffers_;
    std::vector<Write> writes_;   // reused across chunks
    std::shared_ptr<struct UploadState> pending_;   // shared with callbacks in flight
    UploadError failed_ = UploadError::Ok;
};

}  // namespace bllm::residency

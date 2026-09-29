#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <webgpu/webgpu.h>

#include "core/arch/architecture.h"
#include "core/gpu/wgpu_handles.h"
#include "core/residency/plan.h"

namespace bllm::residency {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// Carries out a residency plan. Creates every planned buffer and owns it for
// the life of the loaded model; the KV cache and the kernels refer to these
// buffers by their index in the plan.
//
//   - The file arrives in order, in chunks of any size. A tensor may span
//     chunks.
//   - A tensor's bytes pass through its format's upload transform and its
//     architecture's load transform, where those exist, before they are
//     written.
//   - A candidate duplicate is compared byte for byte with the tensor it may
//     copy. On a match the two share storage and the candidate's own buffer is
//     never created. On a mismatch it is created and written; the plan has
//     already counted it.
//   - Writes are queued. Completion is reported through a callback and never
//     waited for.

enum class UploadError {
    Ok,
    // A chunk did not start where the previous one ended.
    OutOfOrder,
    OutOfMemory,
    DeviceLost,
};

using UploadFinishedFn = void (*)(UploadError error, void* userdata);

class Upload {
public:
    Upload(WGPUDevice device, const ResidencyPlan& plan,
           const arch::Architecture& architecture);

    Upload(const Upload&) = delete;
    Upload& operator=(const Upload&) = delete;

    [[nodiscard]] UploadError write(std::uint64_t file_offset,
                                    std::span<const std::byte> bytes);

    // Invokes `callback` exactly once, after every queued write has completed
    // or failed.
    void finish(UploadFinishedFn callback, void* userdata);

private:
    std::vector<gpu::Buffer> buffers_;
};

}  // namespace bllm::residency

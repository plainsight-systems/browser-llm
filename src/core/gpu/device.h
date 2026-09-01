#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "core/gpu/wgpu_handles.h"

namespace bllm::gpu {

// The limits that constrain how model weights can be laid out on the GPU, and
// how many of them a shader can see at once.
//
// These are four separate WebGPU constraints, not one. Conflating them yields
// a residency plan that uploads successfully and then cannot be bound:
//
//   max_buffer_size                       caps a physical buffer
//   max_storage_buffer_binding_size       caps a *bound range* within one
//   min_storage_buffer_offset_alignment   constrains suballocated offsets
//   max_storage_buffers_per_shader_stage  caps bindings visible to one shader
//
// The last matters more than it looks: weights are uploaded de-interleaved as
// a nibble stream and a scale stream, so each weight tensor costs *two*
// bindings, not one.
struct DeviceLimits {
    std::uint64_t max_buffer_size = 0;
    std::uint64_t max_storage_buffer_binding_size = 0;
    std::uint32_t max_compute_workgroups_per_dimension = 0;
    std::uint32_t max_compute_invocations_per_workgroup = 0;
    std::uint32_t max_storage_buffers_per_shader_stage = 0;
    std::uint32_t min_storage_buffer_offset_alignment = 0;
};

struct AdapterInfo {
    // False when wgpuAdapterGetInfo failed outright. Distinct from an adapter
    // that legitimately reports empty strings for some fields — conflating the
    // two would let a failed query render as a normal, sparsely-populated page.
    bool queried = false;
    std::string vendor;
    std::string architecture;
    std::string device;
    std::string description;
    std::string backend;
};

// Owns a WebGPU instance, adapter and device. Every handle is an RAII alias,
// so release is not written by hand anywhere.
//
// Acquisition is asynchronous because WebGPU's adapter and device requests are
// promises in the browser and this build deliberately does not enable ASYNCIFY.
// The caller supplies a continuation rather than blocking.
//
// Ownership is transferred through a unique_ptr rather than a raw pointer, so
// the callee cannot forget to destroy it (I.11, R.20). On failure the pointer
// is null and `error` describes why. Exactly one of those states is delivered,
// exactly once.
//
// Contract:
//   - `callback` must not be null. It is invoked unconditionally.
//   - `error` is valid only for the duration of the callback. Copy it to keep
//     it; it points into a temporary that dies when the callback returns.
class Device {
public:
    using RequestCallback = void (*)(std::unique_ptr<Device> device,
                                     const char* error,
                                     void* userdata);

    static void request(RequestCallback callback, void* userdata);

    ~Device() = default;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(Device&&) = delete;

    WGPUDevice handle() const { return device_.get(); }
    WGPUQueue queue() const { return queue_.get(); }
    const AdapterInfo& adapter_info() const { return adapter_info_; }

    // Limits of the ACQUIRED DEVICE. These are what validation enforces and
    // what dispatch must be sized against.
    //
    // Not the same thing as adapter_maxima(). WebGPU grants a device the
    // DEFAULT limits unless better ones are named in requiredLimits at
    // acquisition, so an adapter advertising 4 GiB buffers can yield a device
    // capped at 256/128 MiB. Sizing work against the adapter's numbers would
    // pass our own checks and then fail device validation.
    const DeviceLimits& limits() const { return limits_; }

    // What the adapter could grant if asked. Informational only: never size a
    // dispatch or a buffer against these.
    const DeviceLimits& adapter_maxima() const { return adapter_maxima_; }

private:
    Device() = default;
    friend struct PendingDeviceRequest;

    // Declaration order is release order reversed by the compiler: members are
    // destroyed bottom-up, so queue releases before device, device before
    // adapter, adapter before instance.
    Instance instance_;
    Adapter adapter_;
    DeviceHandle device_;
    Queue queue_;
    AdapterInfo adapter_info_;
    DeviceLimits limits_;          // of the acquired device
    DeviceLimits adapter_maxima_;  // of the adapter, informational
};

}  // namespace bllm::gpu

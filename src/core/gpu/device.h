#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "core/gpu/wgpu_handles.h"

namespace bllm::gpu {

// The four limits that constrain how model weights can be laid out on the GPU.
// Recorded so a later performance baseline can name the device it was taken on.
struct DeviceLimits {
    std::uint64_t max_buffer_size = 0;
    std::uint64_t max_storage_buffer_binding_size = 0;
    std::uint32_t max_compute_workgroups_per_dimension = 0;
    std::uint32_t max_compute_invocations_per_workgroup = 0;
};

struct AdapterInfo {
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
    const DeviceLimits& limits() const { return limits_; }

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
    DeviceLimits limits_;
};

}  // namespace bllm::gpu

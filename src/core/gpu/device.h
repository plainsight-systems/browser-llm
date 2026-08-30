#pragma once

#include <cstdint>
#include <string>

#include <webgpu/webgpu.h>

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

// Owns a WebGPU instance, adapter and device.
//
// Acquisition is asynchronous because WebGPU's adapter and device requests are
// promises in the browser and this build deliberately does not enable ASYNCIFY.
// The caller supplies a continuation rather than blocking.
//
// On success `device` is non-null and the callee takes ownership; it must
// `delete` it. On failure `device` is null and `error` describes why. Exactly
// one of those two states is delivered, exactly once.
class Device {
public:
    using RequestCallback = void (*)(Device* device, const char* error, void* userdata);

    static void request(RequestCallback callback, void* userdata);

    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(Device&&) = delete;

    WGPUDevice handle() const { return device_; }
    WGPUQueue queue() const { return queue_; }
    const AdapterInfo& adapter_info() const { return adapter_info_; }
    const DeviceLimits& limits() const { return limits_; }

private:
    Device() = default;

    WGPUInstance instance_ = nullptr;
    WGPUAdapter adapter_ = nullptr;
    WGPUDevice device_ = nullptr;
    WGPUQueue queue_ = nullptr;
    AdapterInfo adapter_info_;
    DeviceLimits limits_;

    struct PendingRequest;
};

}  // namespace bllm::gpu

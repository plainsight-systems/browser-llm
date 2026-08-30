#include "core/gpu/device.h"

#include <cstring>
#include <string>
#include <utility>

namespace bllm::gpu {
namespace {

std::string to_string(WGPUStringView view) {
    if (view.data == nullptr) {
        return {};
    }
    // WGPU_STRLEN means "null-terminated"; anything else is an explicit length.
    return view.length == WGPU_STRLEN ? std::string(view.data)
                                      : std::string(view.data, view.length);
}

const char* backend_name(WGPUBackendType type) {
    switch (type) {
        case WGPUBackendType_WebGPU: return "WebGPU";
        case WGPUBackendType_D3D11: return "D3D11";
        case WGPUBackendType_D3D12: return "D3D12";
        case WGPUBackendType_Metal: return "Metal";
        case WGPUBackendType_Vulkan: return "Vulkan";
        case WGPUBackendType_OpenGL: return "OpenGL";
        case WGPUBackendType_OpenGLES: return "OpenGLES";
        default: return "unknown";
    }
}

}  // namespace

// Carries the caller's continuation across the two async hops. Heap allocated
// because the callbacks outlive the call that started them; freed exactly once
// on whichever path completes.
struct Device::PendingRequest {
    Device* device;
    RequestCallback callback;
    void* userdata;

    void fail(const std::string& message) {
        delete device;
        callback(nullptr, message.c_str(), userdata);
        delete this;
    }

    void succeed() {
        callback(device, nullptr, userdata);
        delete this;
    }
};

Device::~Device() {
    // Reverse acquisition order. Each handle is released exactly once; the
    // class is non-copyable and non-movable so a double release is not
    // representable.
    if (queue_ != nullptr) wgpuQueueRelease(queue_);
    if (device_ != nullptr) wgpuDeviceRelease(device_);
    if (adapter_ != nullptr) wgpuAdapterRelease(adapter_);
    if (instance_ != nullptr) wgpuInstanceRelease(instance_);
}

void Device::request(RequestCallback callback, void* userdata) {
    auto* pending = new PendingRequest{new Device(), callback, userdata};

    pending->device->instance_ = wgpuCreateInstance(nullptr);
    if (pending->device->instance_ == nullptr) {
        pending->fail("could not create a WebGPU instance; this browser may not support WebGPU");
        return;
    }

    WGPURequestAdapterCallbackInfo adapter_cb = {};
    adapter_cb.mode = WGPUCallbackMode_AllowSpontaneous;
    adapter_cb.userdata1 = pending;
    adapter_cb.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
                             WGPUStringView message, void* ud1, void*) {
        auto* p = static_cast<PendingRequest*>(ud1);
        if (status != WGPURequestAdapterStatus_Success || adapter == nullptr) {
            const std::string detail = to_string(message);
            p->fail("no WebGPU adapter available" + (detail.empty() ? "" : ": " + detail));
            return;
        }
        p->device->adapter_ = adapter;

        WGPUAdapterInfo info = {};
        if (wgpuAdapterGetInfo(adapter, &info) == WGPUStatus_Success) {
            p->device->adapter_info_ = AdapterInfo{
                to_string(info.vendor), to_string(info.architecture),
                to_string(info.device), to_string(info.description),
                backend_name(info.backendType)};
            wgpuAdapterInfoFreeMembers(info);
        }

        WGPULimits limits = {};
        if (wgpuAdapterGetLimits(adapter, &limits) == WGPUStatus_Success) {
            p->device->limits_ = DeviceLimits{
                limits.maxBufferSize, limits.maxStorageBufferBindingSize,
                limits.maxComputeWorkgroupsPerDimension,
                limits.maxComputeInvocationsPerWorkgroup};
        }

        WGPURequestDeviceCallbackInfo device_cb = {};
        device_cb.mode = WGPUCallbackMode_AllowSpontaneous;
        device_cb.userdata1 = p;
        device_cb.callback = [](WGPURequestDeviceStatus s, WGPUDevice device,
                                WGPUStringView msg, void* inner_ud1, void*) {
            auto* q = static_cast<PendingRequest*>(inner_ud1);
            if (s != WGPURequestDeviceStatus_Success || device == nullptr) {
                const std::string detail = to_string(msg);
                q->fail("could not acquire a WebGPU device" +
                        (detail.empty() ? "" : ": " + detail));
                return;
            }
            q->device->device_ = device;
            q->device->queue_ = wgpuDeviceGetQueue(device);
            q->succeed();
        };
        wgpuAdapterRequestDevice(adapter, nullptr, device_cb);
    };

    wgpuInstanceRequestAdapter(pending->device->instance_, nullptr, adapter_cb);
}

}  // namespace bllm::gpu

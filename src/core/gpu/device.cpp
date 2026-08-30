#include "core/gpu/device.h"

#include <cstdio>
#include <memory>
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

// WebGPU reports validation and out-of-memory failures here rather than by
// returning null from most creation calls. Without this they are silent, and
// a failed pipeline would surface only as wrong output much later.
void on_uncaptured_error(WGPUDevice const*, WGPUErrorType type,
                         WGPUStringView message, void*, void*) {
    std::fprintf(stderr, "[webgpu] uncaptured error (type %d): %s\n",
                 static_cast<int>(type), to_string(message).c_str());
}

}  // namespace

// Carries the caller's continuation across the two async hops. Heap allocated
// because the callbacks outlive the call that started them; freed exactly once
// on whichever path completes.
struct PendingDeviceRequest {
    std::unique_ptr<Device> device;
    Device::RequestCallback callback;
    void* userdata;

    void fail(const std::string& message) {
        auto* self = this;
        self->callback(nullptr, message.c_str(), self->userdata);
        delete self;
    }

    void succeed() {
        auto* self = this;
        self->callback(std::move(self->device), nullptr, self->userdata);
        delete self;
    }
};

void Device::request(RequestCallback callback, void* userdata) {
    auto* pending = new PendingDeviceRequest{
        std::unique_ptr<Device>(new Device()), callback, userdata};

    pending->device->instance_.reset(wgpuCreateInstance(nullptr));
    if (!pending->device->instance_) {
        pending->fail("could not create a WebGPU instance; this browser may not support WebGPU");
        return;
    }

    WGPURequestAdapterCallbackInfo adapter_cb = {};
    adapter_cb.mode = WGPUCallbackMode_AllowSpontaneous;
    adapter_cb.userdata1 = pending;
    adapter_cb.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
                             WGPUStringView message, void* ud1, void*) {
        auto* p = static_cast<PendingDeviceRequest*>(ud1);
        if (status != WGPURequestAdapterStatus_Success || adapter == nullptr) {
            const std::string detail = to_string(message);
            p->fail("no WebGPU adapter available" + (detail.empty() ? "" : ": " + detail));
            return;
        }
        p->device->adapter_.reset(adapter);

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

        WGPUDeviceDescriptor device_desc = {};
        device_desc.uncapturedErrorCallbackInfo.callback = on_uncaptured_error;

        WGPURequestDeviceCallbackInfo device_cb = {};
        device_cb.mode = WGPUCallbackMode_AllowSpontaneous;
        device_cb.userdata1 = p;
        device_cb.callback = [](WGPURequestDeviceStatus s, WGPUDevice device,
                                WGPUStringView msg, void* inner_ud1, void*) {
            auto* q = static_cast<PendingDeviceRequest*>(inner_ud1);
            if (s != WGPURequestDeviceStatus_Success || device == nullptr) {
                const std::string detail = to_string(msg);
                q->fail("could not acquire a WebGPU device" +
                        (detail.empty() ? "" : ": " + detail));
                return;
            }
            q->device->device_.reset(device);
            q->device->queue_.reset(wgpuDeviceGetQueue(device));
            q->succeed();
        };
        wgpuAdapterRequestDevice(adapter, &device_desc, device_cb);
    };

    wgpuInstanceRequestAdapter(pending->device->instance_.get(), nullptr, adapter_cb);
}

}  // namespace bllm::gpu

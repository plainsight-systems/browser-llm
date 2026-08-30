#include "core/gpu/self_check.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "bllm/shaders_generated.h"
#include "core/gpu/dispatch_math.h"

namespace bllm::gpu {
namespace {

constexpr std::uint32_t kWorkgroupSize = 64;  // must match vector_add.wgsl

WGPUStringView view(std::string_view s) {
    return WGPUStringView{s.data(), s.size()};
}

// Everything the readback callback needs once the encoder work has been
// submitted. Heap allocated: it outlives the call that created it.
struct ReadbackContext {
    WGPUBuffer readback;
    std::vector<float> expected;
    SelfCheckCallback callback;
    void* userdata;

    void finish(SelfCheckResult result) {
        wgpuBufferRelease(readback);
        callback(result, userdata);
        delete this;
    }
};

}  // namespace

void run_self_check(Device& device, std::size_t elements,
                    SelfCheckCallback callback, void* userdata) {
    const auto dispatch = dispatch_count(
        elements, kWorkgroupSize, device.limits().max_compute_workgroups_per_dimension);
    if (dispatch.status != DispatchStatus::Ok) {
        SelfCheckResult r;
        r.error = dispatch.status == DispatchStatus::InvalidWorkgroupSize
                      ? "invalid workgroup size"
                      : "element count exceeds this device's dispatch limit";
        callback(r, userdata);
        return;
    }

    std::vector<float> lhs(elements), rhs(elements), expected(elements);
    for (std::size_t i = 0; i < elements; ++i) {
        lhs[i] = static_cast<float>(i);
        rhs[i] = static_cast<float>(i) * 0.5f;
        expected[i] = lhs[i] + rhs[i];
    }
    const std::uint64_t bytes = static_cast<std::uint64_t>(elements) * sizeof(float);

    WGPUDevice dev = device.handle();

    auto make_buffer = [&](WGPUBufferUsage usage) {
        WGPUBufferDescriptor d = {};
        d.usage = usage;
        d.size = bytes;
        return wgpuDeviceCreateBuffer(dev, &d);
    };
    WGPUBuffer lhs_buf = make_buffer(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer rhs_buf = make_buffer(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer out_buf = make_buffer(WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer readback = make_buffer(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);

    wgpuQueueWriteBuffer(device.queue(), lhs_buf, 0, lhs.data(), bytes);
    wgpuQueueWriteBuffer(device.queue(), rhs_buf, 0, rhs.data(), bytes);

    WGPUShaderSourceWGSL wgsl = {};
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code = view(shaders::vector_add);
    WGPUShaderModuleDescriptor module_desc = {};
    module_desc.nextInChain = &wgsl.chain;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(dev, &module_desc);

    WGPUComputePipelineDescriptor pipeline_desc = {};
    pipeline_desc.compute.module = module;
    pipeline_desc.compute.entryPoint = view("main");
    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(dev, &pipeline_desc);

    WGPUBindGroupEntry entries[3] = {};
    entries[0].binding = 0; entries[0].buffer = lhs_buf; entries[0].size = bytes;
    entries[1].binding = 1; entries[1].buffer = rhs_buf; entries[1].size = bytes;
    entries[2].binding = 2; entries[2].buffer = out_buf; entries[2].size = bytes;
    WGPUBindGroupDescriptor bg_desc = {};
    bg_desc.layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    bg_desc.entryCount = 3;
    bg_desc.entries = entries;
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(dev, &bg_desc);

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(dev, nullptr);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, dispatch.workgroup_count, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, out_buf, 0, readback, 0, bytes);
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(device.queue(), 1, &commands);

    // Everything except the readback buffer is done being referenced by us; the
    // submitted commands hold their own references until they retire.
    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bind_group);
    wgpuBindGroupLayoutRelease(bg_desc.layout);
    wgpuComputePipelineRelease(pipeline);
    wgpuShaderModuleRelease(module);
    wgpuBufferRelease(lhs_buf);
    wgpuBufferRelease(rhs_buf);
    wgpuBufferRelease(out_buf);

    auto* ctx = new ReadbackContext{readback, std::move(expected), callback, userdata};

    WGPUBufferMapCallbackInfo map_cb = {};
    map_cb.mode = WGPUCallbackMode_AllowSpontaneous;
    map_cb.userdata1 = ctx;
    map_cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* ud1, void*) {
        auto* c = static_cast<ReadbackContext*>(ud1);
        SelfCheckResult r;
        r.elements = c->expected.size();

        if (status != WGPUMapAsyncStatus_Success) {
            r.error = "could not map the readback buffer";
            c->finish(r);
            return;
        }
        const auto byte_count = c->expected.size() * sizeof(float);
        const void* mapped = wgpuBufferGetConstMappedRange(c->readback, 0, byte_count);
        if (mapped == nullptr) {
            r.error = "readback buffer mapped but produced no range";
            c->finish(r);
            return;
        }

        const auto* got = static_cast<const float*>(mapped);
        for (std::size_t i = 0; i < c->expected.size(); ++i) {
            if (got[i] != c->expected[i]) {
                ++r.mismatches;
            }
        }
        wgpuBufferUnmap(c->readback);

        r.ok = r.mismatches == 0;
        if (!r.ok) {
            r.error = "GPU result did not match the CPU computation";
        }
        c->finish(r);
    };
    wgpuBufferMapAsync(readback, WGPUMapMode_Read, 0, static_cast<std::size_t>(bytes), map_cb);
}

}  // namespace bllm::gpu

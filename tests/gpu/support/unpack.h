#pragma once

// Test-only: runs a format's unpack on the GPU over stored blocks, as a
// kernel would, and returns what it decoded. The blocks are laid out on the
// device by PieceWriter, as upload lays out a piece, so the test reads what
// upload writes. One invocation per 32-weight group writes its eight vec4s
// out; the output is read back whole. Shader errors fail the test with
// WebGPU's message.

#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <webgpu/webgpu.h>

#include "core/formats/device_layout.h"
#include "core/gpu/callback_mode.h"
#include "core/gpu/device.h"
#include "core/gpu/wgpu_handles.h"
#include "core/residency/piece_writer.h"
#include "core/residency/routes.h"
#include "support/pump.h"

namespace bllm::testing {

// The kernel side of format.h's contract, around the format's unpack.
inline constexpr std::string_view kUnpackHarnessHead = R"(
@group(0) @binding(0) var<storage, read> weights: array<u32>;
)";

inline constexpr std::string_view kUnpackHarnessMain = R"(
struct Params { blocks_in_piece: u32, groups: u32 }
@group(0) @binding(1) var<storage, read_write> decoded: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let group = id.x;
    if (group >= params.groups) { return; }
    let v = unpack(params.blocks_in_piece, group);
    for (var i = 0u; i < 8u; i++) {
        decoded[group * 8u + i] = v[i];
    }
}
)";

// `stored` as one piece of `layout` lays out on the device.
inline std::vector<std::byte> lay_out(const formats::DeviceLayout& layout, std::span<const std::byte> stored) {
    const std::uint64_t blocks = stored.size() / layout.block_bytes;
    const std::uint64_t length = (stored.size() + 3) / 4 * 4;
    const residency::Route route{gguf::TensorId{0}, 0, blocks, &layout, residency::BufferIndex{0}, 0, length};
    residency::PieceWriter writer(std::span(&route, 1), stored.size());
    std::vector<residency::Write> writes;
    REQUIRE(writer.accept(0, stored, writes) == residency::WriteError::Ok);
    REQUIRE(writer.finish(stored.size()) == residency::WriteError::Ok);
    std::vector<std::byte> device(length);
    for (const auto& w : writes) std::memcpy(device.data() + w.offset, w.bytes.data(), w.bytes.size());
    return device;
}

struct ScopeResult {
    WGPUErrorType type = WGPUErrorType_Unknown;
    std::string message;
    bool done = false;
};

inline std::vector<float> run_unpack(WGPUInstance instance, const gpu::Device& device, std::string_view unpack_wgsl,
                                     const formats::DeviceLayout& layout, std::span<const std::byte> stored,
                                     std::uint32_t groups) {
    const auto blocks = static_cast<std::uint32_t>(stored.size() / layout.block_bytes);
    const std::vector<std::byte> on_device = lay_out(layout, stored);
    WGPUDevice dev = device.handle();

    wgpuDevicePushErrorScope(dev, WGPUErrorFilter_Validation);
    const std::string source = std::string(kUnpackHarnessHead) + std::string(unpack_wgsl) +
                               std::string(kUnpackHarnessMain);
    WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgsl.code = WGPUStringView{source.data(), source.size()};
    WGPUShaderModuleDescriptor module_desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    module_desc.nextInChain = &wgsl.chain;
    const gpu::ShaderModule module(wgpuDeviceCreateShaderModule(dev, &module_desc));
    WGPUComputePipelineDescriptor pipeline_desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    pipeline_desc.compute.module = module.get();
    pipeline_desc.compute.entryPoint = WGPUStringView{"main", 4};
    const gpu::ComputePipeline pipeline(wgpuDeviceCreateComputePipeline(dev, &pipeline_desc));

    const auto buffer = [&](WGPUBufferUsage usage, std::uint64_t size) {
        WGPUBufferDescriptor d = WGPU_BUFFER_DESCRIPTOR_INIT;
        d.usage = usage;
        d.size = size;
        return gpu::Buffer(wgpuDeviceCreateBuffer(dev, &d));
    };
    const std::uint64_t out_bytes = std::uint64_t{groups} * 32 * sizeof(float);
    const gpu::Buffer weights = buffer(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, on_device.size());
    const gpu::Buffer decoded = buffer(WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc, out_bytes);
    const gpu::Buffer params = buffer(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, 16);
    const gpu::Buffer readback = buffer(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, out_bytes);
    wgpuQueueWriteBuffer(device.queue(), weights.get(), 0, on_device.data(), on_device.size());
    const std::uint32_t param_words[4] = {blocks, groups, 0, 0};
    wgpuQueueWriteBuffer(device.queue(), params.get(), 0, param_words, sizeof(param_words));

    const gpu::BindGroupLayout bind_layout(wgpuComputePipelineGetBindGroupLayout(pipeline.get(), 0));
    WGPUBindGroupEntry entries[3] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT,
                                     WGPU_BIND_GROUP_ENTRY_INIT};
    entries[0].binding = 0;
    entries[0].buffer = weights.get();
    entries[0].size = on_device.size();
    entries[1].binding = 1;
    entries[1].buffer = decoded.get();
    entries[1].size = out_bytes;
    entries[2].binding = 2;
    entries[2].buffer = params.get();
    entries[2].size = 16;
    WGPUBindGroupDescriptor bind_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bind_desc.layout = bind_layout.get();
    bind_desc.entryCount = 3;
    bind_desc.entries = entries;
    const gpu::BindGroup bind_group(wgpuDeviceCreateBindGroup(dev, &bind_desc));

    const gpu::CommandEncoder encoder(wgpuDeviceCreateCommandEncoder(dev, nullptr));
    {
        const gpu::ComputePassEncoder pass(wgpuCommandEncoderBeginComputePass(encoder.get(), nullptr));
        wgpuComputePassEncoderSetPipeline(pass.get(), pipeline.get());
        wgpuComputePassEncoderSetBindGroup(pass.get(), 0, bind_group.get(), 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass.get(), (groups + 63) / 64, 1, 1);
        wgpuComputePassEncoderEnd(pass.get());
    }
    wgpuCommandEncoderCopyBufferToBuffer(encoder.get(), decoded.get(), 0, readback.get(), 0, out_bytes);
    const gpu::CommandBuffer commands(wgpuCommandEncoderFinish(encoder.get(), nullptr));
    WGPUCommandBuffer raw = commands.get();
    wgpuQueueSubmit(device.queue(), 1, &raw);

    ScopeResult scope;
    WGPUPopErrorScopeCallbackInfo pop = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
    pop.mode = gpu::kCallbackMode;
    pop.userdata1 = &scope;
    pop.callback = [](WGPUPopErrorScopeStatus, WGPUErrorType type, WGPUStringView message, void* userdata, void*) {
        auto& s = *static_cast<ScopeResult*>(userdata);
        s.type = type;
        if (message.data != nullptr) {
            s.message = message.length == WGPU_STRLEN ? std::string(message.data)
                                                      : std::string(message.data, message.length);
        }
        s.done = true;
    };
    wgpuDevicePopErrorScope(dev, pop);
    pump_until(instance, scope.done, "the unpack kernel's scope");
    REQUIRE_MESSAGE(scope.type == WGPUErrorType_NoError, scope.message);

    struct Mapped {
        WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
        bool done = false;
    } mapped;
    WGPUBufferMapCallbackInfo map = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    map.mode = gpu::kCallbackMode;
    map.userdata1 = &mapped;
    map.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata, void*) {
        auto& m = *static_cast<Mapped*>(userdata);
        m.status = status;
        m.done = true;
    };
    wgpuBufferMapAsync(readback.get(), WGPUMapMode_Read, 0, out_bytes, map);
    pump_until(instance, mapped.done, "the decoded weights");
    REQUIRE(mapped.status == WGPUMapAsyncStatus_Success);
    const void* range = wgpuBufferGetConstMappedRange(readback.get(), 0, out_bytes);
    REQUIRE(range != nullptr);
    std::vector<float> out(std::size_t{groups} * 32);
    std::memcpy(out.data(), range, out_bytes);
    wgpuBufferUnmap(readback.get());
    return out;
}

// Bit for bit, but a zero of either sign equals a zero: WGSL may drop a
// zero's sign, and no kernel can tell (format.h).
inline bool same_weight(float got, float want) {
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    std::memcpy(&a, &got, 4);
    std::memcpy(&b, &want, 4);
    return a == b || (got == 0.0f && want == 0.0f);
}

// Every weight the same as the reference's, as same_weight judges.
inline void check_bitwise(std::span<const float> got, std::span<const float> want) {
    REQUIRE(got.size() == want.size());
    std::size_t differing = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!same_weight(got[i], want[i]) && differing++ < 8) {
            CAPTURE(i);
            CHECK(got[i] == want[i]);
        }
    }
    CHECK(differing == 0);
}

}  // namespace bllm::testing

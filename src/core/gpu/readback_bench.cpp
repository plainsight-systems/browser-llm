#include "core/gpu/readback_bench.h"

#include <cstdint>
#include <string_view>
#include <utility>

#include "bllm/shaders_generated.h"
#include "core/gpu/wgpu_handles.h"

namespace bllm::gpu {
namespace {

constexpr std::uint32_t kWorkgroupSize = 64;   // matches vector_add.wgsl
constexpr std::size_t kElements = 64;          // one workgroup; the dispatch is
                                               // not what we are measuring
constexpr std::uint64_t kBytes = kElements * sizeof(float);

WGPUStringView view(std::string_view s) { return WGPUStringView{s.data(), s.size()}; }

// Everything one bench run needs, alive across every async hop.
struct BenchState {
    std::unique_ptr<Device> device;
    std::size_t iterations = 0;
    NowFn now = nullptr;
    BenchCallback callback = nullptr;
    void* userdata = nullptr;

    ShaderModule module;
    ComputePipeline pipeline;
    BindGroupLayout layout;
    Buffer lhs, rhs, out, readback;
    BindGroup bind_group;

    std::size_t done = 0;
    double phase_start = 0.0;
    ReadbackBenchResult result;

    void finish() {
        result.device = std::move(device);
        callback(std::move(result), userdata);
        delete this;
    }

    void fail(std::string message) {
        result.ok = false;
        result.error = std::move(message);
        finish();
    }

    // Encodes one dispatch and copies the output into the readback buffer.
    void submit_one() {
        CommandEncoder encoder(wgpuDeviceCreateCommandEncoder(device->handle(), nullptr));
        {
            ComputePassEncoder pass(
                wgpuCommandEncoderBeginComputePass(encoder.get(), nullptr));
            wgpuComputePassEncoderSetPipeline(pass.get(), pipeline.get());
            wgpuComputePassEncoderSetBindGroup(pass.get(), 0, bind_group.get(), 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass.get(), 1, 1, 1);
            wgpuComputePassEncoderEnd(pass.get());
        }
        wgpuCommandEncoderCopyBufferToBuffer(encoder.get(), out.get(), 0,
                                             readback.get(), 0, kBytes);
        CommandBuffer commands(wgpuCommandEncoderFinish(encoder.get(), nullptr));
        WGPUCommandBuffer raw = commands.get();
        wgpuQueueSubmit(device->queue(), 1, &raw);
    }
};

void start_sequential_iteration(BenchState* s);

void on_mapped(WGPUMapAsyncStatus status, WGPUStringView, void* ud1, void*) {
    auto* s = static_cast<BenchState*>(ud1);
    if (status != WGPUMapAsyncStatus_Success) {
        s->fail("readback map failed during the sequential phase");
        return;
    }
    // Touch the data: a map whose result is never read could in principle be
    // optimised differently, and the decode loop genuinely reads its token.
    const void* mapped = wgpuBufferGetConstMappedRange(s->readback.get(), 0,
                                                       static_cast<std::size_t>(kBytes));
    volatile float first = mapped != nullptr ? static_cast<const float*>(mapped)[0] : 0.0f;
    (void)first;
    wgpuBufferUnmap(s->readback.get());

    s->result.sequential_ms.push_back(s->now() - s->phase_start);
    ++s->done;

    if (s->done < s->iterations) {
        start_sequential_iteration(s);
        return;
    }

    // Phase two: submit every dispatch, then read back once.
    s->phase_start = s->now();
    for (std::size_t i = 0; i < s->iterations; ++i) {
        s->submit_one();
    }
    WGPUBufferMapCallbackInfo info = {};
    info.mode = WGPUCallbackMode_AllowSpontaneous;
    info.userdata1 = s;
    info.callback = [](WGPUMapAsyncStatus st, WGPUStringView, void* u1, void*) {
        auto* b = static_cast<BenchState*>(u1);
        if (st != WGPUMapAsyncStatus_Success) {
            b->fail("readback map failed during the batched phase");
            return;
        }
        wgpuBufferUnmap(b->readback.get());
        b->result.batched_total_ms = b->now() - b->phase_start;
        b->result.ok = true;
        b->finish();
    };
    wgpuBufferMapAsync(s->readback.get(), WGPUMapMode_Read, 0,
                       static_cast<std::size_t>(kBytes), info);
}

void start_sequential_iteration(BenchState* s) {
    s->phase_start = s->now();
    s->submit_one();

    WGPUBufferMapCallbackInfo info = {};
    info.mode = WGPUCallbackMode_AllowSpontaneous;
    info.userdata1 = s;
    info.callback = on_mapped;
    wgpuBufferMapAsync(s->readback.get(), WGPUMapMode_Read, 0,
                       static_cast<std::size_t>(kBytes), info);
}

}  // namespace

void run_readback_bench(std::unique_ptr<Device> device, std::size_t iterations,
                        NowFn now, BenchCallback callback, void* userdata) {
    ReadbackBenchResult early;
    if (device == nullptr || now == nullptr || iterations == 0) {
        early.error = "readback bench requires a device, a clock, and a non-zero count";
        early.device = std::move(device);
        callback(std::move(early), userdata);
        return;
    }

    auto* s = new BenchState{};
    s->device = std::move(device);
    s->iterations = iterations;
    s->now = now;
    s->callback = callback;
    s->userdata = userdata;
    s->result.iterations = iterations;
    s->result.sequential_ms.reserve(iterations);

    WGPUDevice dev = s->device->handle();
    auto make = [&](WGPUBufferUsage usage) {
        WGPUBufferDescriptor d = {};
        d.usage = usage;
        d.size = kBytes;
        return Buffer(wgpuDeviceCreateBuffer(dev, &d));
    };
    s->lhs = make(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    s->rhs = make(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    s->out = make(WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    s->readback = make(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    if (!s->lhs || !s->rhs || !s->out || !s->readback) {
        s->fail("could not allocate bench buffers");
        return;
    }

    WGPUShaderSourceWGSL wgsl = {};
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code = view(shaders::vector_add);
    WGPUShaderModuleDescriptor md = {};
    md.nextInChain = &wgsl.chain;
    s->module.reset(wgpuDeviceCreateShaderModule(dev, &md));

    WGPUComputePipelineDescriptor pd = {};
    pd.compute.module = s->module.get();
    pd.compute.entryPoint = view("main");
    s->pipeline.reset(wgpuDeviceCreateComputePipeline(dev, &pd));
    s->layout.reset(wgpuComputePipelineGetBindGroupLayout(s->pipeline.get(), 0));

    WGPUBindGroupEntry entries[3] = {};
    entries[0].binding = 0; entries[0].buffer = s->lhs.get(); entries[0].size = kBytes;
    entries[1].binding = 1; entries[1].buffer = s->rhs.get(); entries[1].size = kBytes;
    entries[2].binding = 2; entries[2].buffer = s->out.get(); entries[2].size = kBytes;
    WGPUBindGroupDescriptor bd = {};
    bd.layout = s->layout.get();
    bd.entryCount = 3;
    bd.entries = entries;
    s->bind_group.reset(wgpuDeviceCreateBindGroup(dev, &bd));

    start_sequential_iteration(s);
}

}  // namespace bllm::gpu

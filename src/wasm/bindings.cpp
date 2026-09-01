// The only Emscripten-aware translation unit in this repository.
//
// Its job is translation, not behavior: it starts the device request, forwards
// the result to JavaScript as JSON, and owns nothing else. Product behavior
// belongs in src/core.

#include <emscripten.h>
#include <emscripten/eventloop.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "core/gpu/device.h"
#include "core/run_guard.h"
#include "core/gpu/self_check.h"
#include "core/gpu/readback_bench.h"

namespace {

// Element count for the toolchain self-check. Large enough to span many
// workgroups at the shader's 64-wide group, small enough to stay far inside
// any device's buffer limits.
constexpr std::size_t kSelfCheckElements = 4096;

// A device that has not answered in this long is not going to. WebGPU gives
// no cancellation, so this bounds how long the module can appear busy, not
// how long the request actually runs.
constexpr double kRunTimeoutMs = 15000.0;

// Round trips measured by the spike. Enough for a stable median.
constexpr std::size_t kBenchIterations = 200;

// Generations ride through WebGPU's void* userdata. A 32-bit generation is
// used so this holds on wasm32, where a pointer is 4 bytes.
static_assert(sizeof(std::uint32_t) <= sizeof(void*),
              "generation must fit in a userdata pointer");

void* to_userdata(std::uint32_t generation) {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(generation));
}

std::uint32_t to_generation(void* userdata) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(userdata));
}

// Escapes only what a JSON string requires. Adapter descriptions come from the
// driver, so they are not assumed to be free of quotes or backslashes.
std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (const char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[7];
                    // Cast before formatting: a plain char would sign-extend
                    // if this branch ever widened past the control range.
                    std::snprintf(buf, sizeof(buf), "\\u%04x",
                                  static_cast<unsigned>(static_cast<unsigned char>(c)));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// Delivers a JSON result to the page. Defined in JS because the page owns
// presentation; C++ owns only what happened.
EM_JS(void, bllm_deliver, (const char* json), {
    const text = UTF8ToString(json);
    if (typeof globalThis.bllmOnResult === 'function') {
        globalThis.bllmOnResult(JSON.parse(text));
    } else {
        console.error('bllm: no result handler registered', text);
    }
});

void report_failure(const std::string& stage, const std::string& error) {
    const std::string json = std::string("{\"ok\":false,\"stage\":\"") + stage +
                             "\",\"error\":\"" + json_escape(error) + "\"}";
    bllm_deliver(json.c_str());
}

// Serialises runs and identifies late callbacks. Logic lives in core and is
// tested natively; this file only supplies the clock.
bllm::RunGuard& guard() {
    static bllm::RunGuard g;
    return g;
}

std::uint32_t& pending_generation() {
    static std::uint32_t g = 0;
    return g;
}

int& timeout_id() {
    static int id = 0;
    return id;
}

void disarm_timeout() {
    if (timeout_id() != 0) {
        emscripten_clear_timeout(timeout_id());
        timeout_id() = 0;
    }
}

// The device arrives inside the result and is released when it goes out of
// scope here, so this reports the device it actually measured rather than
// re-reading whatever is current.
void on_self_check(bllm::gpu::SelfCheckResult result, void* userdata) {
    // A run already closed by the timeout must not report a second result.
    // The device still arrives here and is released as this returns.
    if (!guard().complete(to_generation(userdata))) {
        return;
    }
    disarm_timeout();

    if (!result.ok) {
        report_failure("self_check", result.error);
        return;
    }
    if (result.device == nullptr) {
        report_failure("self_check", "internal: result carried no device");
        return;
    }
    const auto& device = *result.device;
    const auto& info = device.adapter_info();
    const auto& limits = device.limits();
    const auto& maxima = device.adapter_maxima();

    std::string json = "{\"ok\":true,\"adapter\":{";
    json += "\"vendor\":\"" + json_escape(info.vendor) + "\",";
    json += "\"architecture\":\"" + json_escape(info.architecture) + "\",";
    json += "\"device\":\"" + json_escape(info.device) + "\",";
    json += "\"description\":\"" + json_escape(info.description) + "\",";
    json += "\"backend\":\"" + json_escape(info.backend) + "\",";
    json += "\"queried\":" + std::string(info.queried ? "true" : "false") +
            "},\"limits\":{";
    json += "\"maxBufferSize\":" + std::to_string(limits.max_buffer_size) + ",";
    json += "\"maxStorageBufferBindingSize\":" +
            std::to_string(limits.max_storage_buffer_binding_size) + ",";
    json += "\"maxComputeWorkgroupsPerDimension\":" +
            std::to_string(limits.max_compute_workgroups_per_dimension) + ",";
    json += "\"maxComputeInvocationsPerWorkgroup\":" +
            std::to_string(limits.max_compute_invocations_per_workgroup) + ",";
    json += "\"maxStorageBuffersPerShaderStage\":" +
            std::to_string(limits.max_storage_buffers_per_shader_stage) + ",";
    json += "\"minStorageBufferOffsetAlignment\":" +
            std::to_string(limits.min_storage_buffer_offset_alignment) + "},";
    json += "\"adapterMaxima\":{";
    json += "\"maxBufferSize\":" + std::to_string(maxima.max_buffer_size) + ",";
    json += "\"maxStorageBufferBindingSize\":" +
            std::to_string(maxima.max_storage_buffer_binding_size) + "},";
    json += "\"selfCheck\":{\"elements\":" + std::to_string(result.elements) +
            ",\"mismatches\":" + std::to_string(result.mismatches) + "}}";
    bllm_deliver(json.c_str());
}

// --- readback measurement spike -------------------------------------------
// Answers one question before the decode loop is designed: what does a
// serialized GPU round trip cost? Run on request only; not on the normal path.

double now_ms() { return emscripten_get_now(); }

void on_bench(bllm::gpu::ReadbackBenchResult result, void*) {
    if (!guard().complete(pending_generation())) {
        return;
    }
    disarm_timeout();

    if (!result.ok) {
        report_failure("bench", result.error);
        return;
    }
    auto seq = result.sequential_ms;
    std::sort(seq.begin(), seq.end());
    const auto pick = [&seq](double q) {
        if (seq.empty()) return 0.0;
        const auto i = static_cast<std::size_t>(q * static_cast<double>(seq.size() - 1));
        return seq[i];
    };
    double total = 0.0;
    for (const double v : seq) total += v;

    std::string json = "{\"ok\":true,\"bench\":{";
    json += "\"iterations\":" + std::to_string(result.iterations) + ",";
    json += "\"seqMinMs\":" + std::to_string(pick(0.0)) + ",";
    json += "\"seqMedianMs\":" + std::to_string(pick(0.5)) + ",";
    json += "\"seqP95Ms\":" + std::to_string(pick(0.95)) + ",";
    json += "\"seqMaxMs\":" + std::to_string(pick(1.0)) + ",";
    json += "\"seqMeanMs\":" +
            std::to_string(seq.empty() ? 0.0 : total / static_cast<double>(seq.size())) + ",";
    json += "\"batchedTotalMs\":" + std::to_string(result.batched_total_ms) + "}}";
    bllm_deliver(json.c_str());
}

void on_device_for_bench(std::unique_ptr<bllm::gpu::Device> device, const char* error,
                         void* userdata) {
    const std::uint32_t generation = to_generation(userdata);
    if (device == nullptr) {
        if (guard().complete(generation)) {
            disarm_timeout();
            report_failure("device", error != nullptr ? error : "unknown error");
        }
        return;
    }
    if (!guard().active() || guard().generation() != generation) {
        return;
    }
    bllm::gpu::run_readback_bench(std::move(device), kBenchIterations, now_ms,
                                  on_bench, nullptr);
}

void on_device(std::unique_ptr<bllm::gpu::Device> device, const char* error,
               void* userdata) {
    const std::uint32_t generation = to_generation(userdata);

    if (device == nullptr) {
        if (guard().complete(generation)) {
            disarm_timeout();
            report_failure("device", error != nullptr ? error : "unknown error");
        }
        return;
    }
    // A device that arrives after the run timed out is released here rather
    // than starting work nobody is waiting for.
    if (!guard().active() || guard().generation() != generation) {
        return;
    }
    // Ownership passes into the check and comes back in the result.
    bllm::gpu::run_self_check(std::move(device), kSelfCheckElements,
                              on_self_check, to_userdata(generation));
}

void on_timeout(void* userdata) {
    timeout_id() = 0;
    if (!guard().complete(to_generation(userdata))) {
        return;   // the run already finished; nothing to report
    }
    report_failure("timeout", "the GPU did not respond within " +
                                  std::to_string(static_cast<int>(kRunTimeoutMs / 1000)) +
                                  " seconds; the request may still be pending");
}

}  // namespace

extern "C" {

// Entry point called from the worker once the module is instantiated.
EMSCRIPTEN_KEEPALIVE void bllm_run_self_check() {
    // One run at a time. A second would acquire another device while one is
    // still live and report two results the page cannot tell apart.
    const std::uint32_t generation = guard().begin();
    if (generation == bllm::RunGuard::kNoRun) {
        report_failure("request", "a self-check is already running");
        return;
    }
    // Armed before the request, so a request that never calls back is still
    // bounded. WebGPU cannot be cancelled, so this frees the module rather
    // than the GPU.
    pending_generation() = generation;
    timeout_id() = emscripten_set_timeout(on_timeout, kRunTimeoutMs,
                                          to_userdata(generation));
    bllm::gpu::Device::request(on_device, to_userdata(generation));
}

// Measurement spike. Deliberately a separate entry point so the normal path
// is unchanged and cannot accidentally run it.
EMSCRIPTEN_KEEPALIVE void bllm_run_readback_bench() {
    const std::uint32_t generation = guard().begin();
    if (generation == bllm::RunGuard::kNoRun) {
        report_failure("request", "a run is already in progress");
        return;
    }
    pending_generation() = generation;
    timeout_id() = emscripten_set_timeout(on_timeout, kRunTimeoutMs,
                                          to_userdata(generation));
    bllm::gpu::Device::request(on_device_for_bench, to_userdata(generation));
}

}  // extern "C"

int main() {
    // Nothing runs at load. The page decides when to start, so a failure is
    // attributable to an explicit request rather than to module instantiation.
    return 0;
}

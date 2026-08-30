// The only Emscripten-aware translation unit in this repository.
//
// Its job is translation, not behavior: it starts the device request, forwards
// the result to JavaScript as JSON, and owns nothing else. Product behavior
// belongs in src/core.

#include <emscripten.h>

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "core/gpu/device.h"
#include "core/gpu/self_check.h"

namespace {

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

// Construct-on-first-use rather than a namespace-scope global (R.6, I.2,
// LIFE.6). The device outlives the request that created it because the page
// keeps using it; ownership is explicit and single.
std::unique_ptr<bllm::gpu::Device>& active_device() {
    static std::unique_ptr<bllm::gpu::Device> device;
    return device;
}

bool& request_in_flight() {
    static bool in_flight = false;
    return in_flight;
}

void on_self_check(const bllm::gpu::SelfCheckResult& result, void*) {
    request_in_flight() = false;

    if (!result.ok) {
        report_failure("self_check", result.error);
        return;
    }
    const auto& device = *active_device();
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
            std::to_string(limits.max_compute_invocations_per_workgroup) + "},";
    json += "\"adapterMaxima\":{";
    json += "\"maxBufferSize\":" + std::to_string(maxima.max_buffer_size) + ",";
    json += "\"maxStorageBufferBindingSize\":" +
            std::to_string(maxima.max_storage_buffer_binding_size) + "},";
    json += "\"selfCheck\":{\"elements\":" + std::to_string(result.elements) +
            ",\"mismatches\":" + std::to_string(result.mismatches) + "}}";
    bllm_deliver(json.c_str());
}

void on_device(std::unique_ptr<bllm::gpu::Device> device, const char* error, void*) {
    if (device == nullptr) {
        request_in_flight() = false;
        report_failure("device", error != nullptr ? error : "unknown error");
        return;
    }
    // Replacing any previous device destroys it here rather than leaking it.
    active_device() = std::move(device);
    bllm::gpu::run_self_check(*active_device(), 4096, on_self_check, nullptr);
}

}  // namespace

extern "C" {

// Entry point called from the worker once the module is instantiated.
EMSCRIPTEN_KEEPALIVE void bllm_run_self_check() {
    // Re-entry would overwrite the device while a readback still referenced
    // it. Refuse explicitly rather than racing.
    if (request_in_flight()) {
        report_failure("request", "a self-check is already running");
        return;
    }
    request_in_flight() = true;
    bllm::gpu::Device::request(on_device, nullptr);
}

}  // extern "C"

int main() {
    // Nothing runs at load. The page decides when to start, so a failure is
    // attributable to an explicit request rather than to module instantiation.
    return 0;
}

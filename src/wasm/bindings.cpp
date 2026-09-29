// The only Emscripten-aware translation unit in this repository.
//
// Its job is translation, not behavior: it starts the device request, forwards
// the result to JavaScript as JSON, and owns nothing else. Product behavior
// belongs in src/core.
//
// Contract 11, the boundary: this file and web/worker.js are the only two
// places JavaScript and C++ meet. The crossings are preflight a header prefix,
// load a chunk of the file, generate from a rendered prompt and the turn's
// policy, and cancel. Text goes back one crossing per token (WASM.2).

#include <emscripten.h>
#include <emscripten/eventloop.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "core/gguf/reader.h"
#include "core/gpu/device.h"
#include "core/preflight/preflight.h"
#include "core/run_guard.h"
#include "core/gpu/self_check.h"
#include "core/diagnostics.h"
#if BLLM_DIAGNOSTICS_ENABLED
#include "core/gpu/readback_bench.h"
#endif

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

// Answers one request from the worker. Requests carry an id so an answer
// that arrives later, from a callback, still reaches the request it belongs to.
EM_JS(void, bllm_reply, (std::uint32_t request, const char* json), {
    globalThis.bllmOnReply(request, JSON.parse(UTF8ToString(json)));
});

std::string json_string(std::string_view text) {
    return "\"" + json_escape(std::string(text)) + "\"";
}

// A token's text, looked up by the id stored under `id_key`, as JSON; null if
// the file declares no such token.
std::string token_json(bllm::gguf::ByteSource& source, const bllm::gguf::TensorIndex& index,
                       std::string_view id_key) {
    std::uint32_t id = 0;
    bllm::gguf::ArrayLocation tokens{};
    std::string text;
    if (index.read_u32(id_key, id) != bllm::gguf::MetadataError::Ok ||
        index.read_array("tokenizer.ggml.tokens", tokens) != bllm::gguf::MetadataError::Ok ||
        bllm::gguf::read_string_element(source, tokens, id, text).error !=
            bllm::gguf::ReadError::Ok) {
        return "null";
    }
    return json_string(text);
}

// What the page needs to render a conversation for this model: the chat
// template the file carries, and the text of the tokens it refers to.
std::string chat_json(bllm::gguf::ByteSource& source, const bllm::gguf::TensorIndex& index) {
    std::string_view chat_template;
    const bool has_template = index.read_string("tokenizer.chat_template", chat_template) ==
                              bllm::gguf::MetadataError::Ok;
    return "{\"template\":" + (has_template ? json_string(chat_template) : std::string("null")) +
           ",\"bosToken\":" + token_json(source, index, "tokenizer.ggml.bos_token_id") +
           ",\"eosToken\":" + token_json(source, index, "tokenizer.ggml.eos_token_id") + "}";
}

// The preflight answer: bytes the reader still needs, a file that cannot be
// read, or the verdict on a file that can.
std::string preflight_json(bllm::gguf::ByteSource& source, const bllm::gguf::ReadResult& read,
                           const bllm::gguf::TensorIndex& index) {
    using bllm::gguf::ReadError;
    if (read.error == ReadError::NeedMoreBytes) {
        return "{\"status\":\"need-bytes\",\"bytesNeeded\":" +
               std::to_string(read.bytes_needed) + "}";
    }
    if (read.error != ReadError::Ok) {
        return "{\"status\":\"unreadable\",\"error\":" +
               json_string(bllm::gguf::to_string(read.error)) + "}";
    }
    // The device-fit gate fails closed in this build (preflight.cpp) and reads
    // neither limits nor policy, so none are passed.
    const auto verdict = bllm::preflight::preflight(index, bllm::residency::DeviceLimits{},
                                                    bllm::policy::LoadPolicy{});
    std::string_view architecture;
    const bool named = index.read_string("general.architecture", architecture) ==
                       bllm::gguf::MetadataError::Ok;

    std::string json = "{\"status\":\"read\",\"architecture\":";
    json += named ? json_string(architecture) : "null";
    json += ",\"tensorCount\":" + std::to_string(index.tensors().size());
    json += ",\"chat\":" + chat_json(source, index);
    json += ",\"accepted\":" + std::string(verdict.accepted() ? "true" : "false");
    json += ",\"rejections\":[";
    for (std::size_t i = 0; i < verdict.rejections.size(); ++i) {
        const auto& r = verdict.rejections[i];
        json += i == 0 ? "" : ",";
        json += "{\"gate\":" + json_string(bllm::preflight::to_string(r.gate)) +
                ",\"detail\":" + json_string(r.detail) + "}";
    }
    return json + "]}";
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

#if BLLM_DIAGNOSTICS_ENABLED
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

#endif  // BLLM_DIAGNOSTICS_ENABLED

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

// Reads the index from the front of a model file and answers with the
// preflight verdict. `resident` bytes are the file's first bytes; `file_size`
// is the whole file's length, as a double because JavaScript numbers are
// exact to 2^53 and no model file approaches that.
EMSCRIPTEN_KEEPALIVE void bllm_preflight(std::uint32_t request, const std::byte* resident,
                                         std::uint32_t resident_length, double file_size) {
    // Checked before the conversion: casting a NaN, a negative or a
    // fractional double to an integer is undefined or lossy, and the value
    // comes from outside C++. The resident prefix cannot exceed the file.
    constexpr double kMaxExactInteger = 9007199254740992.0;   // 2^53
    if (!(file_size >= resident_length && file_size <= kMaxExactInteger) ||
        file_size != static_cast<double>(static_cast<std::uint64_t>(file_size))) {
        bllm_reply(request, "{\"status\":\"unreadable\",\"error\":\"the file size is not valid\"}");
        return;
    }
    bllm::gguf::MemoryByteSource source{{resident, resident_length},
                                        static_cast<std::uint64_t>(file_size)};
    bllm::gguf::TensorIndex index;
    const auto read = bllm::gguf::read_index(source, index);
    bllm_reply(request, preflight_json(source, read, index).c_str());
}

#if BLLM_DIAGNOSTICS_ENABLED
// Measurement spike. Present only in a diagnostic build; the clean build does
// not compile it, so the symbol is absent from the shipped module.
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
#endif  // BLLM_DIAGNOSTICS_ENABLED

}  // extern "C"

int main() {
    // Nothing runs at load. The page decides when to start, so a failure is
    // attributable to an explicit request rather than to module instantiation.
    return 0;
}

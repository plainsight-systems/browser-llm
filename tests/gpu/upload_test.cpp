// Upload on Dawn: the cases upload.h names, against a real device. What
// lands where is piece_writer's to prove (tests/residency_piece_writer_test);
// these prove what finish may claim.

#include <doctest/doctest.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/arch/architecture.h"
#include "core/capability/capability.h"
#include "core/formats/format.h"
#include "core/gguf/reader.h"
#include "core/gpu/device.h"
#include "core/gpu/wgpu_handles.h"
#include "core/residency/plan.h"
#include "core/residency/upload.h"
#include "support/acquire.h"
#include "support/gguf_fixture.h"
#include "support/pump.h"

using namespace bllm;
using residency::Upload;
using residency::UploadError;
using testing::acquire;
using testing::pump_until;

namespace {

// Formats of the test's own: the capability table lists none until a
// format's unpack exists.
constexpr formats::Format kF32{formats::kF32Layout, "fn unpack_f32() {}"};
constexpr formats::Format kQ4_0{formats::kQ4_0Layout, "fn unpack_q4_0() {}"};

const formats::Format* both(gguf::TensorType type) noexcept {
    if (type == gguf::TensorType::F32) return &kF32;
    if (type == gguf::TensorType::Q4_0) return &kQ4_0;
    return nullptr;
}

const formats::Format* f32_only(gguf::TensorType type) noexcept {
    return type == gguf::TensorType::F32 ? &kF32 : nullptr;
}

struct Model {
    std::vector<std::byte> bytes;
    gguf::TensorIndex index;
    residency::ResidencyPlan plan;
};

Model load(const std::string& fixture) {
    Model m;
    m.bytes = testing::load_gguf_fixture(fixture);
    gguf::MemoryByteSource source{m.bytes};
    REQUIRE(gguf::read_index(source, m.index).error == gguf::ReadError::Ok);
    std::string_view name;
    REQUIRE(m.index.read_string("general.architecture", name) == gguf::MetadataError::Ok);
    model::ModelDescription description;
    REQUIRE(capability::find_architecture(name)->describe(m.index, description).ok());
    (void)residency::plan_residency(m.index, description, residency::DeviceLimits{256ull << 20, 128ull << 20, 256},
                                    policy::LoadPolicy{}, m.plan);
    return m;
}

constexpr std::size_t kChunk = 256;

struct Ready {
    std::unique_ptr<Upload> upload;
    UploadError error = UploadError::Internal;
    std::string subject;
    bool done = false;
};

void on_ready(std::unique_ptr<Upload> upload, UploadError error, const char* subject, void* userdata) {
    auto& r = *static_cast<Ready*>(userdata);
    r.upload = std::move(upload);
    r.error = error;
    r.subject = subject;
    r.done = true;
}

std::unique_ptr<Upload> begin(const gpu::Device& device, const Model& m) {
    Ready ready;
    Upload::begin(device, m.index, m.plan, m.bytes.size(), both, {}, kChunk, on_ready, &ready);
    pump_until(device.instance(), ready.done, "the buffers");
    REQUIRE_MESSAGE(ready.error == UploadError::Ok, ready.subject);
    REQUIRE(ready.upload != nullptr);
    return std::move(ready.upload);
}

// Counts every call, so "exactly once" is checked, not assumed.
struct Reported {
    UploadError error = UploadError::Internal;
    int calls = 0;
    bool done = false;
};

void on_reported(UploadError error, void* userdata) {
    auto& r = *static_cast<Reported*>(userdata);
    r.error = error;
    ++r.calls;
    r.done = true;
}

// Streams the whole file, sending each chunk only once the last is accepted,
// as the page does.
void stream(WGPUInstance instance, Upload& upload, const Model& m) {
    for (std::size_t at = 0; at < m.bytes.size(); at += kChunk) {
        Reported accepted;
        const auto chunk = std::span(m.bytes).subspan(at, std::min(kChunk, m.bytes.size() - at));
        upload.write(at, chunk, on_reported, &accepted);
        pump_until(instance, accepted.done, "a chunk's acceptance");
        REQUIRE(accepted.calls == 1);
        REQUIRE(accepted.error == UploadError::Ok);
    }
}

UploadError finish(WGPUInstance instance, Upload& upload) {
    Reported finished;
    upload.finish(on_reported, &finished);
    pump_until(instance, finished.done, "finish");
    CHECK(finished.calls == 1);
    return finished.error;
}

}  // namespace

TEST_CASE("an upload on a live device finishes Ok, its witness read back") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    // F32 only, and F32 with a Q4_0 weight, whose streams are rearranged.
    for (const char* fixture : {"tiny_qwen3", "tiny_qwen3_odd_blocks"}) {
        CAPTURE(fixture);
        const Model m = load(fixture);
        auto upload = begin(*device, m);
        CHECK(!upload->routes().empty());
        CHECK(upload->buffer(residency::BufferIndex{0}) != nullptr);
        stream(instance.get(), *upload, m);
        CHECK(finish(instance.get(), *upload) == UploadError::Ok);
    }
}

TEST_CASE("the first chunk is accepted at once; a later one once the chunk before it is done") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Model m = load("tiny_qwen3");
    auto upload = begin(*device, m);

    Reported first;
    upload->write(0, std::span(m.bytes).first(kChunk), on_reported, &first);
    CHECK(first.calls == 1);   // before any event is processed
    CHECK(first.error == UploadError::Ok);

    Reported second;
    upload->write(kChunk, std::span(m.bytes).subspan(kChunk, kChunk), on_reported, &second);
    CHECK(second.calls == 0);   // waits on the first chunk's writes
    pump_until(instance.get(), second.done, "the second chunk's acceptance");
    CHECK(second.calls == 1);
    CHECK(second.error == UploadError::Ok);
}

TEST_CASE("a write the device rejects makes finish Validation, never Ok") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Model m = load("tiny_qwen3");
    auto upload = begin(*device, m);
    // A destroyed buffer refuses every write to it.
    wgpuBufferDestroy(upload->buffer(residency::BufferIndex{0}));
    stream(instance.get(), *upload, m);
    CHECK(finish(instance.get(), *upload) == UploadError::Validation);
}

TEST_CASE("finish after the device is destroyed is DeviceLost or Unconfirmed, never Ok") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Model m = load("tiny_qwen3");
    auto upload = begin(*device, m);
    stream(instance.get(), *upload, m);
    wgpuDeviceDestroy(device->handle());
    const UploadError e = finish(instance.get(), *upload);
    CAPTURE(static_cast<int>(e));
    CHECK((e == UploadError::DeviceLost || e == UploadError::Unconfirmed));
}

TEST_CASE("destroying the Upload with finish pending reports Cancelled, once") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Model m = load("tiny_qwen3");
    auto upload = begin(*device, m);
    stream(instance.get(), *upload, m);
    Reported finished;
    upload->finish(on_reported, &finished);
    upload.reset();
    pump_until(instance.get(), finished.done, "the cancelled finish");
    // Let anything else that would call it again arrive.
    for (int i = 0; i < 100; ++i) wgpuInstanceProcessEvents(instance.get());
    CHECK(finished.calls == 1);
    CHECK(finished.error == UploadError::Cancelled);
}

TEST_CASE("releasing the caller's device and instance with work pending changes nothing") {
    const Model m = load("tiny_qwen3");
    gpu::Instance instance{wgpuCreateInstance(nullptr)};
    auto device = acquire(instance.get());
    auto upload = begin(*device, m);
    stream(instance.get(), *upload, m);
    Reported finished;
    upload->finish(on_reported, &finished);
    device.reset();
    instance = gpu::Instance{};
    // The Upload's own instance reference keeps its callbacks alive.
    pump_until(upload->instance(), finished.done, "finish with the caller's references gone");
    CHECK(finished.calls == 1);
    CHECK(finished.error == UploadError::Ok);
}

TEST_CASE("a tensor whose format is not listed is refused by name before any buffer is made") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Model m = load("tiny_qwen3_odd_blocks");
    Ready ready;
    Upload::begin(*device, m.index, m.plan, m.bytes.size(), f32_only, {}, kChunk, on_ready, &ready);
    CHECK(ready.done);   // decided without the device
    CHECK(ready.upload == nullptr);
    CHECK(ready.error == UploadError::UnsupportedFormat);
    CHECK(ready.subject == "extra.weight");
}

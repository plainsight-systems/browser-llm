// The upload check on Dawn: every byte upload wrote, read back and compared.
// Built only with diagnostics, as the check is.

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "core/gpu/device.h"
#include "core/gpu/wgpu_handles.h"
#include "core/residency/piece_writer.h"
#include "core/residency/upload.h"
#include "core/residency/upload_check.h"
#include "support/acquire.h"
#include "support/pump.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;
using residency::CheckError;
using residency::UploadCheck;

namespace {

struct Checked {
    CheckError error = CheckError::Internal;
    int calls = 0;
    bool done = false;
};

void on_checked(CheckError error, void* userdata) {
    auto& c = *static_cast<Checked*>(userdata);
    c.error = error;
    ++c.calls;
    c.done = true;
}

// Streams the file through the check as the page does, stopping at the first
// chunk refused; returns its error.
CheckError check_all(WGPUInstance instance, UploadCheck& check, const Model& m) {
    for (std::size_t at = 0; at < m.bytes.size(); at += kChunk) {
        Checked accepted;
        check.check(at, std::span(m.bytes).subspan(at, std::min(kChunk, m.bytes.size() - at)), on_checked, &accepted);
        pump_until(instance, accepted.done, "a chunk compared");
        REQUIRE(accepted.calls == 1);
        if (accepted.error != CheckError::Ok) return accepted.error;
    }
    return check.finish(m.bytes.size());
}

// An upload, finished Ok, ready to be checked.
std::unique_ptr<Upload> uploaded(WGPUInstance instance, const gpu::Device& device, const Model& m) {
    auto upload = begin(device, m);
    REQUIRE(stream(instance, *upload, m) == UploadError::Ok);
    REQUIRE(finish(instance, *upload) == UploadError::Ok);
    return upload;
}

// The file's first chunk is its header, which no route reads, so it is
// accepted at once. A test that needs a comparison pending sends the whole
// file as one chunk, to a check sized for it.
std::span<const std::byte> whole(const Model& m) { return m.bytes; }

}  // namespace

TEST_CASE("every byte upload wrote is found on the device where the plan put it") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    for (const char* fixture : {"tiny_qwen3", "tiny_qwen3_odd_blocks"}) {
        CAPTURE(fixture);
        const Model m = load(fixture);
        const auto upload = uploaded(instance.get(), *device, m);
        UploadCheck check(*upload, kChunk);
        CHECK(check_all(instance.get(), check, m) == CheckError::Ok);
        CHECK(check.mismatches().empty());
    }
}

TEST_CASE("one word changed on the device is found, by buffer, offset and tensor") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Model m = load("tiny_qwen3_odd_blocks");
    const auto upload = uploaded(instance.get(), *device, m);

    // The Q4_0 weight: its device bytes start with its first block's first
    // stream, 16 bytes wide, so its second word is that block's bytes 4 to 8
    // of the stream — inside a write, not at its start.
    const auto routes = upload->routes();
    const auto route = std::find_if(routes.begin(), routes.end(), [&](const residency::Route& r) {
        return upload->tensor_name(r.tensor) == "extra.weight";
    });
    REQUIRE(route != routes.end());
    REQUIRE(route->layout->streams[0].width >= 8);
    const std::size_t first = route->file_offset + route->layout->streams[0].offset + 4;
    std::array<std::byte, 4> changed{};
    for (std::size_t i = 0; i < changed.size(); ++i) changed[i] = ~m.bytes[first + i];
    wgpuQueueWriteBuffer(device->queue(), upload->buffer(route->buffer), route->buffer_offset + 4, changed.data(),
                         changed.size());

    UploadCheck check(*upload, kChunk);
    CHECK(check_all(instance.get(), check, m) == CheckError::Ok);
    REQUIRE(check.mismatches().size() == 1);
    const auto& found = check.mismatches()[0];
    CHECK(found.buffer == route->buffer);
    CHECK(found.offset == route->buffer_offset + 4);
    CHECK(found.tensor == "extra.weight");
}

TEST_CASE("a word changed in every piece is found in each, though one write covers many") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Model m = load("tiny_qwen3_odd_blocks");
    const auto upload = uploaded(instance.get(), *device, m);

    // The whole file as one chunk, so pieces share writes: what upload
    // wrote, regenerated as the check does.
    residency::PieceWriter writer(upload->routes(), m.bytes.size());
    std::vector<residency::Write> writes;
    REQUIRE(writer.accept(0, m.bytes, writes) == residency::WriteError::Ok);

    // Each piece's second word, complemented; a piece of one word is left
    // alone.
    std::vector<const residency::Route*> changed;
    for (const residency::Route& r : upload->routes()) {
        if (r.length < 8) continue;
        const auto w = std::find_if(writes.begin(), writes.end(), [&](const residency::Write& w) {
            return w.buffer == r.buffer && w.offset <= r.buffer_offset && r.buffer_offset + 8 <= w.offset + w.bytes.size();
        });
        REQUIRE(w != writes.end());
        std::array<std::byte, 4> word{};
        for (std::size_t i = 0; i < word.size(); ++i) word[i] = ~w->bytes[r.buffer_offset + 4 - w->offset + i];
        wgpuQueueWriteBuffer(device->queue(), upload->buffer(r.buffer), r.buffer_offset + 4, word.data(), word.size());
        changed.push_back(&r);
    }
    REQUIRE(writes.size() < changed.size());   // pieces share writes

    UploadCheck check(*upload, m.bytes.size());
    Checked accepted;
    check.check(0, whole(m), on_checked, &accepted);
    pump_until(instance.get(), accepted.done, "the file compared");
    REQUIRE(accepted.error == CheckError::Ok);
    CHECK(check.finish(m.bytes.size()) == CheckError::Ok);
    REQUIRE(check.mismatches().size() == changed.size());
    for (std::size_t i = 0; i < changed.size(); ++i) {
        CAPTURE(i);
        CHECK(check.mismatches()[i].buffer == changed[i]->buffer);
        CHECK(check.mismatches()[i].offset == changed[i]->buffer_offset + 4);
        CHECK(check.mismatches()[i].tensor == upload->tensor_name(changed[i]->tensor));
    }
}

TEST_CASE("a check on a destroyed device fails; it never ends with nothing compared") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Model m = load("tiny_qwen3");
    const auto upload = uploaded(instance.get(), *device, m);
    UploadCheck check(*upload, kChunk);
    wgpuDeviceDestroy(device->handle());
    const CheckError e = check_all(instance.get(), check, m);
    CAPTURE(static_cast<int>(e));
    CHECK((e == CheckError::DeviceLost || e == CheckError::MapFailed));
}

TEST_CASE("destroying the check with a comparison pending reports Cancelled, once") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Model m = load("tiny_qwen3");
    const auto upload = uploaded(instance.get(), *device, m);
    auto check = std::make_unique<UploadCheck>(*upload, m.bytes.size());
    Checked accepted;
    check->check(0, whole(m), on_checked, &accepted);
    REQUIRE(accepted.calls == 0);   // a comparison is pending
    check.reset();
    pump_until(instance.get(), accepted.done, "the cancelled comparison");
    for (int i = 0; i < 100; ++i) wgpuInstanceProcessEvents(instance.get());
    CHECK(accepted.calls == 1);
    CHECK(accepted.error == CheckError::Cancelled);
}

TEST_CASE("releasing the caller's device and instance with a comparison pending changes nothing") {
    const Model m = load("tiny_qwen3");
    gpu::Instance instance{wgpuCreateInstance(nullptr)};
    auto device = acquire(instance.get());
    const auto upload = uploaded(instance.get(), *device, m);
    UploadCheck check(*upload, m.bytes.size());
    Checked accepted;
    check.check(0, whole(m), on_checked, &accepted);
    REQUIRE(accepted.calls == 0);   // a comparison is pending
    device.reset();
    instance = gpu::Instance{};
    pump_until(upload->instance(), accepted.done, "the comparison with the caller's references gone");
    CHECK(accepted.calls == 1);
    CHECK(accepted.error == CheckError::Ok);
}

TEST_CASE("finish with a chunk still being compared, or the file short, is Unfinished") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Model m = load("tiny_qwen3");
    const auto upload = uploaded(instance.get(), *device, m);

    UploadCheck pending(*upload, m.bytes.size());
    Checked accepted;
    pending.check(0, whole(m), on_checked, &accepted);
    REQUIRE(accepted.calls == 0);   // every route reached, but not yet compared
    CHECK(pending.finish(m.bytes.size()) == CheckError::Unfinished);
    pump_until(instance.get(), accepted.done, "the comparison");

    UploadCheck shortened(*upload, kChunk);
    Checked first;
    shortened.check(0, std::span(m.bytes).first(kChunk), on_checked, &first);
    pump_until(instance.get(), first.done, "the first chunk compared");
    CHECK(shortened.finish(m.bytes.size()) == CheckError::Unfinished);
}

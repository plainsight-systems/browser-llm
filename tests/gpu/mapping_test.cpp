// The mapping and witness classifiers, row by row, without a device: the
// tables in mapping.h and upload.h are what these check.

#include <doctest/doctest.h>

#include <array>
#include <cstddef>
#include <span>

#include "core/residency/mapping.h"
#include "core/residency/upload.h"

using bllm::gpu::DeviceStatus;
using bllm::residency::mapping_result;
using bllm::residency::Mapped;
using bllm::residency::UploadError;
using bllm::residency::witness_result;

namespace {

const DeviceStatus kLive{};
const DeviceStatus kLost{true, WGPUDeviceLostReason_Destroyed, "destroyed"};

}  // namespace

TEST_CASE("mapping_result: every status, with and without a range, on a live and a lost device") {
    struct Row {
        WGPUMapAsyncStatus status;
        bool has_range;
        const DeviceStatus* device;
        Mapped want;
    };
    const auto other = static_cast<WGPUMapAsyncStatus>(0x7F);
    const Row rows[] = {
        {WGPUMapAsyncStatus_Success, true, &kLive, Mapped::Ok},
        {WGPUMapAsyncStatus_Success, true, &kLost, Mapped::Ok},
        {WGPUMapAsyncStatus_Success, false, &kLive, Mapped::Internal},
        {WGPUMapAsyncStatus_CallbackCancelled, true, &kLive, Mapped::Cancelled},
        {WGPUMapAsyncStatus_CallbackCancelled, false, &kLost, Mapped::Cancelled},
        {WGPUMapAsyncStatus_Error, false, &kLive, Mapped::Internal},
        {WGPUMapAsyncStatus_Error, false, &kLost, Mapped::Internal},
        {WGPUMapAsyncStatus_Aborted, false, &kLost, Mapped::DeviceLost},
        {WGPUMapAsyncStatus_Aborted, false, &kLive, Mapped::Unexplained},
        {other, true, &kLive, Mapped::Internal},
        {other, false, &kLost, Mapped::Internal},
    };
    for (const Row& r : rows) {
        CAPTURE(static_cast<int>(r.status));
        CAPTURE(r.has_range);
        CAPTURE(r.device->lost);
        CHECK(mapping_result(r.status, r.has_range, *r.device) == r.want);
    }
}

TEST_CASE("witness_result: only a completed mapping holding the expected bytes is Ok") {
    const std::array expected{std::byte{0xC4}, std::byte{0x0F}, std::byte{0x2A}, std::byte{0x91}};
    const std::span<const std::byte, 4> want(expected);
    auto other = expected;
    other[3] = std::byte{0x90};

    CHECK(witness_result(Mapped::Ok, expected, want) == UploadError::Ok);
    CHECK(witness_result(Mapped::Ok, other, want) == UploadError::WitnessMismatch);
    CHECK(witness_result(Mapped::Ok, std::span(expected).first(3), want) == UploadError::WitnessMismatch);
    CHECK(witness_result(Mapped::Ok, {}, want) == UploadError::WitnessMismatch);

    // A mapping that did not complete is its cause, whatever bytes are passed.
    for (const auto bytes : {std::span<const std::byte>(expected), std::span<const std::byte>()}) {
        CHECK(witness_result(Mapped::Cancelled, bytes, want) == UploadError::Cancelled);
        CHECK(witness_result(Mapped::Internal, bytes, want) == UploadError::Internal);
        CHECK(witness_result(Mapped::DeviceLost, bytes, want) == UploadError::DeviceLost);
        CHECK(witness_result(Mapped::Unexplained, bytes, want) == UploadError::Unconfirmed);
    }
}

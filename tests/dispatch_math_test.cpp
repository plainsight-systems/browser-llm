#include <doctest/doctest.h>

#include <cstdint>
#include <limits>

#include "core/gpu/dispatch_math.h"

using bllm::gpu::dispatch_count;
using bllm::gpu::DispatchStatus;

namespace {
constexpr std::uint32_t kLimit = 65535;  // WebGPU's guaranteed minimum.
}

TEST_CASE("exact multiples need no partial workgroup") {
    CHECK(dispatch_count(64, 64, kLimit).workgroup_count == 1);
    CHECK(dispatch_count(128, 64, kLimit).workgroup_count == 2);
    CHECK(dispatch_count(6400, 64, kLimit).workgroup_count == 100);
}

TEST_CASE("non-multiples round up so the tail is covered") {
    CHECK(dispatch_count(1, 64, kLimit).workgroup_count == 1);
    CHECK(dispatch_count(63, 64, kLimit).workgroup_count == 1);
    CHECK(dispatch_count(65, 64, kLimit).workgroup_count == 2);
    CHECK(dispatch_count(129, 64, kLimit).workgroup_count == 3);
}

TEST_CASE("zero elements is a valid empty dispatch") {
    const auto r = dispatch_count(0, 64, kLimit);
    CHECK(r.status == DispatchStatus::Ok);
    CHECK(r.workgroup_count == 0);
}

TEST_CASE("a zero workgroup size is rejected, not divided by") {
    const auto r = dispatch_count(100, 0, kLimit);
    CHECK(r.status == DispatchStatus::InvalidWorkgroupSize);
    CHECK(r.workgroup_count == 0);
}

TEST_CASE("the device limit is inclusive at the boundary") {
    const std::uint64_t exactly_at_limit =
        static_cast<std::uint64_t>(kLimit) * 64;

    const auto ok = dispatch_count(exactly_at_limit, 64, kLimit);
    CHECK(ok.status == DispatchStatus::Ok);
    CHECK(ok.workgroup_count == kLimit);

    const auto over = dispatch_count(exactly_at_limit + 1, 64, kLimit);
    CHECK(over.status == DispatchStatus::ExceedsDeviceLimit);
}

TEST_CASE("oversized workloads fail explicitly rather than truncating") {
    // Silently clamping here would leave most of the data unprocessed and the
    // kernel would return a plausible but wrong result.
    const auto r = dispatch_count(1'000'000'000ull, 64, kLimit);
    CHECK(r.status == DispatchStatus::ExceedsDeviceLimit);
    CHECK(r.workgroup_count == 0);
}

TEST_CASE("element counts near the top of the range do not overflow") {
    const std::uint64_t huge = std::numeric_limits<std::uint64_t>::max();

    // Would overflow if implemented as (n + wg - 1) / wg.
    const auto r = dispatch_count(huge, 64, kLimit);
    CHECK(r.status == DispatchStatus::ExceedsDeviceLimit);

    // And with a limit large enough to accept it, the arithmetic still holds.
    const auto wide =
        dispatch_count(huge, 64, std::numeric_limits<std::uint32_t>::max());
    CHECK(wide.status == DispatchStatus::ExceedsDeviceLimit);
}

TEST_CASE("a workgroup size of one maps elements to workgroups directly") {
    CHECK(dispatch_count(1000, 1, kLimit).workgroup_count == 1000);
}

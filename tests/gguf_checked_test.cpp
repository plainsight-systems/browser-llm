#include <doctest/doctest.h>

#include <cstdint>
#include <limits>

#include "core/gguf/checked.h"

using namespace bllm::gguf;
constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();

TEST_CASE("checked_add refuses the wrap instead of producing a small number") {
    std::uint64_t out = 0;
    CHECK(checked_add(2, 3, out));
    CHECK(out == 5);

    CHECK(checked_add(kMax, 0, out));
    CHECK(out == kMax);

    // The case that matters: a hostile offset plus a plausible length.
    out = 12345;
    CHECK_FALSE(checked_add(kMax, 1, out));
    CHECK(out == 12345);          // untouched on failure
    CHECK_FALSE(checked_add(kMax - 5, 10, out));
}

TEST_CASE("checked_mul refuses shape products that do not fit") {
    std::uint64_t out = 0;
    CHECK(checked_mul(1024, 1024, out));
    CHECK(out == 1048576);

    CHECK(checked_mul(0, kMax, out));      // zero is not an overflow
    CHECK(out == 0);

    CHECK_FALSE(checked_mul(kMax, 2, out));
    CHECK_FALSE(checked_mul(1ull << 32, 1ull << 32, out));
}

TEST_CASE("range_within is correct at the boundary") {
    CHECK(range_within(0, 100, 100));      // exactly fills
    CHECK(range_within(100, 0, 100));      // empty range at the end
    CHECK(range_within(0, 0, 0));
    CHECK_FALSE(range_within(0, 101, 100));
    CHECK_FALSE(range_within(100, 1, 100));
    CHECK_FALSE(range_within(101, 0, 100)); // offset itself past the end
}

TEST_CASE("range_within cannot be defeated by a wrapping offset") {
    // The naive form `offset + length <= total` wraps to a small number here
    // and reports the range as in-bounds. The subtraction form does not.
    CHECK_FALSE(range_within(kMax, 10, 1000));
    CHECK_FALSE(range_within(kMax - 4, 10, 1000));

    // Demonstrate the defect the implementation avoids.
    const std::uint64_t naive = kMax + 10;   // wraps to 9
    CHECK(naive < 1000);                     // ...and would pass a naive check
}

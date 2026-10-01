#include <doctest/doctest.h>

#include <string>

#include "support/test_data.h"

TEST_CASE("fetched test data is the pinned file") {
    // tools/fetch_test_data.py verified its SHA-256; this proves the tests
    // read from where it was put.
    const std::string text = bllm::testing::load_test_data("unicode/NormalizationTest-16.0.0.txt");
    CHECK(text.starts_with("# NormalizationTest-16.0.0.txt"));
}

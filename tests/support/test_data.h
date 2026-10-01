#pragma once

#include <fstream>
#include <iterator>
#include <string>

// Reads a file of fetched test data — TEST SUPPORT ONLY.
//
// Fetched data is listed, with its SHA-256, in tests/fixtures/external.json
// and downloaded by tools/fetch_test_data.py (make test-data). A missing file
// is a hard failure, never a skipped test: a test that quietly skips when its
// data is absent passes for the wrong reason.
namespace bllm::testing {

[[nodiscard]] inline std::string load_test_data(const std::string& name) {
    const std::string path = std::string(BLLM_TEST_DATA_DIR) + "/" + name;
    std::ifstream in(path, std::ios::binary);
    REQUIRE_MESSAGE(in.is_open(), "missing test data: " << path << " (run make test-data)");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

}  // namespace bllm::testing

#include <doctest/doctest.h>

#include <fstream>
#include <sstream>
#include <string>

#include "bllm/shaders_generated.h"

namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE_MESSAGE(in.is_open(), "could not open " << path);
    std::ostringstream buf;
    buf << in.rdbuf();
    return buf.str();
}

}  // namespace

// The embedded shader and the .wgsl on disk must be the same bytes. If build
// codegen ever silently diverges from source, every kernel built on it is
// suspect, so this is checked rather than assumed.
TEST_CASE("embedded vector_add matches the shader source byte for byte") {
    const std::string on_disk =
        read_file(std::string(BLLM_SHADER_DIR) + "/vector_add.wgsl");

    CHECK(std::string(bllm::shaders::vector_add) == on_disk);
}

TEST_CASE("the embedded shader is not empty and declares an entry point") {
    const std::string_view src = bllm::shaders::vector_add;
    CHECK(src.size() > 0);
    CHECK(src.find("@compute") != std::string_view::npos);
    CHECK(src.find("fn main") != std::string_view::npos);
}

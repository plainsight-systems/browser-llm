#pragma once

#include <cstddef>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

// Loads a generated GGUF fixture by case name — TEST SUPPORT ONLY.
//
// Every fixture is produced by tools/make_fixture_gguf.py, so a reviewer can
// see exactly what each byte is. A missing fixture is a hard failure rather
// than an empty buffer: an empty buffer would make a reader test pass for the
// wrong reason.
//
// The caller owns the returned bytes and must keep them alive for as long as
// any MemoryByteSource over them, which is why this returns by value rather
// than handing back a span into a temporary.
namespace bllm::testing {

[[nodiscard]] inline std::vector<std::byte> load_gguf_fixture(const std::string& name) {
    const std::string path = std::string(BLLM_GGUF_FIXTURE_DIR) + "/" + name + ".gguf";
    std::ifstream in(path, std::ios::binary);
    REQUIRE_MESSAGE(in.is_open(), "missing fixture: " << path);
    const std::vector<char> raw((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
    std::vector<std::byte> bytes(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        bytes[i] = static_cast<std::byte>(static_cast<unsigned char>(raw[i]));
    }
    return bytes;
}

}  // namespace bllm::testing

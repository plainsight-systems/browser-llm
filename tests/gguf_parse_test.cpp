#include <doctest/doctest.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include "core/gguf/reader.h"

using namespace bllm::gguf;

namespace {

std::vector<std::byte> load(const std::string& name) {
    const std::string path = std::string(BLLM_GGUF_FIXTURE_DIR) + "/" + name + ".gguf";
    std::ifstream in(path, std::ios::binary);
    REQUIRE_MESSAGE(in.is_open(), "missing fixture: " << path);
    std::vector<char> raw((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
    std::vector<std::byte> out(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        out[i] = static_cast<std::byte>(static_cast<unsigned char>(raw[i]));
    }
    return out;
}

// Parses a fixture and returns the error, keeping the reader alive for
// inspection via the callback.
template <typename F>
ReadError with_reader(const std::string& name, F&& inspect) {
    const auto bytes = load(name);
    MemoryByteSource source{bytes};
    Reader reader{source};
    const auto err = reader.parse();
    inspect(reader, err);
    return err;
}

}  // namespace

TEST_CASE("a valid file yields the expected index") {
    with_reader("valid", [](const Reader& r, ReadError err) {
        REQUIRE(err == ReadError::Ok);

        REQUIRE(r.tensors().size() == 2);
        const auto& embd = r.tensors()[0];
        CHECK(embd.name == "token_embd.weight");
        CHECK(embd.type == TensorType::Q4_0);
        CHECK(embd.dimension_count == 2);
        CHECK(embd.dimensions[0] == 64);
        CHECK(embd.dimensions[1] == 2);
        CHECK(embd.element_count == 128);
        // 128 elements = 4 blocks = 4 * 18 bytes.
        CHECK(embd.data_length == 72);
        CHECK(embd.is_quantized());

        const auto& norm = r.tensors()[1];
        CHECK(norm.name == "output_norm.weight");
        CHECK(norm.type == TensorType::F32);
        CHECK(norm.element_count == 4);
        CHECK(norm.data_length == 16);
        CHECK_FALSE(norm.is_quantized());
    });
}

TEST_CASE("metadata is indexed by location, not materialised") {
    with_reader("valid", [](const Reader& r, ReadError err) {
        REQUIRE(err == ReadError::Ok);
        CHECK(r.metadata().size() == 5);

        const auto* arch = r.find("general.architecture");
        REQUIRE(arch != nullptr);
        CHECK(arch->type == ValueType::String);
        CHECK(arch->value_length > 0);

        // The token array is located and skipped, not decoded.
        const auto* tokens = r.find("tokenizer.ggml.tokens");
        REQUIRE(tokens != nullptr);
        CHECK(tokens->type == ValueType::Array);

        CHECK(r.find("does.not.exist") == nullptr);
    });
}

TEST_CASE("declared alignment is honoured and tensor data starts inside the file") {
    with_reader("valid", [](const Reader& r, ReadError err) {
        REQUIRE(err == ReadError::Ok);
        CHECK(r.alignment() == 32);
        CHECK(r.tensor_data_start() % r.alignment() == 0);
        CHECK(r.tensor_data_start() > 0);
    });
}

TEST_CASE("tensor offsets are absolute and every region lies inside the file") {
    const auto bytes = load("valid");
    MemoryByteSource source{bytes};
    Reader reader{source};
    REQUIRE(reader.parse() == ReadError::Ok);

    for (const auto& t : reader.tensors()) {
        CHECK(t.data_offset >= reader.tensor_data_start());
        CHECK(t.data_offset + t.data_length <= source.size());
    }
}

#include <doctest/doctest.h>

#include <cstddef>
#include <string>
#include <vector>

#include "core/gguf/reader.h"
#include "support/gguf_fixture.h"

using namespace bllm::gguf;

namespace {

using bllm::testing::load_gguf_fixture;

// Parses a fixture and returns the error, keeping the reader alive for
// inspection via the callback.
template <typename F>
ReadError with_reader(const std::string& name, F&& inspect) {
    const auto bytes = load_gguf_fixture(name);
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
    const auto bytes = load_gguf_fixture("valid");
    MemoryByteSource source{bytes};
    Reader reader{source};
    REQUIRE(reader.parse() == ReadError::Ok);

    for (const auto& t : reader.tensors()) {
        CHECK(t.data_offset >= reader.tensor_data_start());
        CHECK(t.data_offset + t.data_length <= source.size());
    }
}

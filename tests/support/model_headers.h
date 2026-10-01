#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/gguf/reader.h"
#include "support/test_data.h"

// The listed models' real headers, fetched as test data — TEST SUPPORT ONLY.
// tools/make_tokenizer_fixtures.py pins each header and writes headers.inc;
// tools/fetch_test_data.py fetches them (make test-data).
namespace bllm::testing {

struct ModelHeader {
    std::string_view model;       // its id in web/models.json
    std::string_view data_name;   // the fetched header, under the test data
    std::uint64_t file_size;      // the whole file's size, which reading the index checks against
};

inline constexpr ModelHeader kModelHeaders[] = {
#include "fixtures/tokenizer/headers.inc"
};

// A model's header, read. A source over `bytes` is made where it is needed,
// as MemoryByteSource{std::as_bytes(std::span{bytes}), file_size}; the index
// holds no reference to the bytes.
struct ReadHeader {
    std::string bytes;
    std::uint64_t file_size;
    gguf::TensorIndex index;
};

[[nodiscard]] inline ReadHeader read_model_header(std::string_view model) {
    for (const ModelHeader& h : kModelHeaders) {
        if (h.model != model) continue;
        ReadHeader out{load_test_data(std::string(h.data_name)), h.file_size, {}};
        gguf::MemoryByteSource source{std::as_bytes(std::span{out.bytes}), out.file_size};
        REQUIRE(gguf::read_index(source, out.index).error == gguf::ReadError::Ok);
        return out;
    }
    FAIL("no header for model " << model);
    return {};
}

}  // namespace bllm::testing

#include <doctest/doctest.h>

#include <algorithm>
#include <string>

#include "core/gguf/reader.h"
#include "core/preflight/preflight.h"
#include "support/gguf_fixture.h"

using namespace bllm;

namespace {

preflight::Verdict preflight_fixture(const std::string& name) {
    const auto bytes = testing::load_gguf_fixture(name);
    gguf::MemoryByteSource source{bytes};
    gguf::TensorIndex index;
    REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);
    return preflight::preflight(index, residency::DeviceLimits{}, policy::LoadPolicy{});
}

bool has_rejection(const preflight::Verdict& verdict, preflight::Gate gate,
                   const std::string& detail) {
    return std::any_of(verdict.rejections.begin(), verdict.rejections.end(),
                       [&](const preflight::Rejection& r) {
                           return r.gate == gate && r.detail == detail;
                       });
}

}  // namespace

TEST_CASE("every gate that can run reports, and names what it rejects") {
    // The fixture names a real architecture and uses two real formats, none
    // of which this build implements, and declares no tokenizer.
    const auto verdict = preflight_fixture("valid");
    using preflight::Gate;

    CHECK_FALSE(verdict.accepted());
    CHECK(has_rejection(verdict, Gate::Architecture, "architecture \"qwen3\" is not supported"));
    CHECK(has_rejection(verdict, Gate::Formats,
                        "format Q4_0 is not supported (1 tensor, first token_embd.weight)"));
    CHECK(has_rejection(verdict, Gate::Formats,
                        "format F32 is not supported (1 tensor, first output_norm.weight)"));
    CHECK(has_rejection(verdict, Gate::Tokenizer,
                        "the file does not declare tokenizer.ggml.model"));
    // Fit needs the architecture's description, so it does not run.
    CHECK(std::none_of(verdict.rejections.begin(), verdict.rejections.end(),
                       [](const auto& r) { return r.gate == Gate::DeviceFit; }));
}

TEST_CASE("an unsupported format is reported once, counting every tensor that uses it") {
    const auto verdict = preflight_fixture("shared_format");
    const auto formats = std::count_if(verdict.rejections.begin(), verdict.rejections.end(),
                                       [](const auto& r) {
                                           return r.gate == preflight::Gate::Formats;
                                       });
    CHECK(formats == 2);
    CHECK(has_rejection(verdict, preflight::Gate::Formats,
                        "format Q4_0 is not supported (2 tensors, first first.weight)"));
}

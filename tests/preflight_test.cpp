#include <doctest/doctest.h>

#include <algorithm>
#include <string>

#include "core/gguf/reader.h"
#include "core/preflight/preflight.h"
#include "support/gguf_fixture.h"

using namespace bllm;
using preflight::Blocker;
using preflight::Stage;
using preflight::Verdict;

namespace {

Verdict preflight_fixture(const std::string& name) {
    const auto bytes = testing::load_gguf_fixture(name);
    gguf::MemoryByteSource source{bytes};
    gguf::TensorIndex index;
    REQUIRE(gguf::read_index(source, index).error == gguf::ReadError::Ok);
    return preflight::preflight(index, residency::DeviceLimits{}, policy::LoadPolicy{});
}

bool blocked(const Verdict& verdict, Stage stage, const std::string& detail) {
    return std::any_of(verdict.blockers.begin(), verdict.blockers.end(),
                       [&](const Blocker& b) { return b.stage == stage && b.detail == detail; });
}

}  // namespace

TEST_CASE("the stage reached is the one before the earliest stage blocked") {
    CHECK(Verdict{}.reached() == Stage::Run);
    CHECK(Verdict{{{Stage::Run, "r"}}}.reached() == Stage::Upload);
    CHECK(Verdict{{{Stage::Run, "r"}, {Stage::Describe, "d"}, {Stage::Fit, "f"}}}.reached() ==
          Stage::Download);
}

TEST_CASE("a readable file can always be downloaded, whatever this build can run") {
    for (const char* name : {"valid", "no_architecture", "architecture_not_string",
                             "tokenizer_named", "shared_format", "q6_k_tensor"}) {
        CAPTURE(name);
        CHECK(preflight_fixture(name).reached() >= Stage::Download);
    }
}

TEST_CASE("every check reports, each naming the stage it stops") {
    // The fixture names a real architecture and uses two real formats, none
    // of which this build implements, and declares no tokenizer.
    const auto verdict = preflight_fixture("valid");

    CHECK(verdict.reached() == Stage::Download);
    CHECK(blocked(verdict, Stage::Describe, "architecture \"qwen3\" is not supported"));
    CHECK(blocked(verdict, Stage::Run,
                  "format Q4_0 is not supported (1 tensor, first token_embd.weight)"));
    CHECK(blocked(verdict, Stage::Run,
                  "format F32 is not supported (1 tensor, first output_norm.weight)"));
    CHECK(blocked(verdict, Stage::Run, "the file does not declare tokenizer.ggml.model"));
}

TEST_CASE("every stage past the last one this build implements is blocked by name") {
    const auto verdict = preflight_fixture("valid");
    for (int s = static_cast<int>(preflight::kImplementedThrough) + 1;
         s <= static_cast<int>(Stage::Run); ++s) {
        const auto stage = static_cast<Stage>(s);
        CAPTURE(preflight::to_string(stage));
        CHECK(blocked(verdict, stage,
                      "the " + std::string(preflight::to_string(stage)) +
                          " stage is not implemented in this build"));
    }
}

TEST_CASE("an unsupported format is reported once, counting every tensor that uses it") {
    const auto verdict = preflight_fixture("shared_format");
    const auto formats = std::count_if(verdict.blockers.begin(), verdict.blockers.end(),
                                       [](const Blocker& b) {
                                           return b.detail.rfind("format ", 0) == 0;
                                       });
    CHECK(formats == 2);
    CHECK(blocked(verdict, Stage::Run, "format Q4_0 is not supported (2 tensors, first first.weight)"));
}

TEST_CASE("a file that names no architecture, or names it wrongly, says which") {
    CHECK(blocked(preflight_fixture("no_architecture"), Stage::Describe,
                  "the file does not declare general.architecture"));
    CHECK(blocked(preflight_fixture("architecture_not_string"), Stage::Describe,
                  "general.architecture is not a string"));
}

TEST_CASE("the tokenizer and the pre-tokenizer are judged separately, each by name") {
    const auto verdict = preflight_fixture("tokenizer_named");
    CHECK(blocked(verdict, Stage::Run, "tokenizer \"gpt2\" is not supported"));
    CHECK(blocked(verdict, Stage::Run, "pre-tokenizer \"qwen2\" is not supported"));
}

TEST_CASE("no blocker names a stage the reader alone decides, and every one says something") {
    for (const char* name : {"valid", "no_architecture", "architecture_not_string",
                             "tokenizer_named", "shared_format"}) {
        CAPTURE(name);
        for (const Blocker& b : preflight_fixture(name).blockers) {
            CHECK(b.stage > Stage::Download);
            CHECK_FALSE(b.detail.empty());
        }
    }
}

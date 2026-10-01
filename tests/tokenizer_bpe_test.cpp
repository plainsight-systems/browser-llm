#include <doctest/doctest.h>

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "core/tokenizer/bpe/byte_level_bpe.h"
#include "core/tokenizer/bpe/piece_cache.h"
#include "core/tokenizer/pretokenize.h"
#include "support/model_headers.h"

using namespace bllm;
using namespace bllm::tokenizer;

namespace {

struct EncodeCase {
    std::string_view name;
    std::string_view text;
    std::vector<std::uint32_t> ids;
};

// Each text and the token IDs Hugging Face tokenizers and llama.cpp both
// encode it to, or the one DECISIONS follows (tools/make_tokenizer_fixtures.py).
const EncodeCase kQwen3Cases[] = {
#include "fixtures/tokenizer/qwen3-0.6b-q4_0.inc"
};
const EncodeCase kLlamaCases[] = {
#include "fixtures/tokenizer/llama-3.2-1b-instruct-q4_0.inc"
};

bpe::ByteLevelBpe load(std::string_view model, const PreTokenizer& pretokenizer) {
    const auto header = bllm::testing::read_model_header(model);
    gguf::MemoryByteSource source{std::as_bytes(std::span{header.bytes}), header.file_size};
    bpe::ByteLevelBpe bpe;
    const auto r = bpe::load_byte_level_bpe(source, header.index, pretokenizer, bpe);
    REQUIRE_MESSAGE(r.ok(), r.subject);
    return bpe;
}

std::vector<std::uint32_t> encode(const bpe::ByteLevelBpe& bpe, std::string_view text) {
    std::vector<TokenId> tokens;
    REQUIRE(bpe.encode(text, tokens) == EncodeError::Ok);
    std::vector<std::uint32_t> ids;
    for (const TokenId t : tokens) ids.push_back(static_cast<std::uint32_t>(t));
    return ids;
}

std::vector<std::uint32_t> encode(const bpe::ByteLevelBpe& bpe, std::string_view text, bpe::PieceCache& cache) {
    std::vector<TokenId> tokens;
    REQUIRE(bpe.encode(text, tokens, cache) == EncodeError::Ok);
    std::vector<std::uint32_t> ids;
    for (const TokenId t : tokens) ids.push_back(static_cast<std::uint32_t>(t));
    return ids;
}

// Every case without a cache, then three times through one cache (cold, then
// warm), then through a one-slot cache whose every piece evicts the last.
void check_cases(const bpe::ByteLevelBpe& bpe, std::span<const EncodeCase> cases) {
    bpe::PieceCache cache;
    bpe::PieceCache tiny{1};
    for (const EncodeCase& c : cases) {
        CAPTURE(c.name);
        CHECK(encode(bpe, c.text) == c.ids);
        for (int pass = 0; pass < 3; ++pass) CHECK(encode(bpe, c.text, cache) == c.ids);
        CHECK(encode(bpe, c.text, tiny) == c.ids);
    }
}

}  // namespace

TEST_CASE("Qwen3 encodes every fixture text to the references' token IDs") {
    check_cases(load("qwen3-0.6b-q4_0", kQwen2), kQwen3Cases);
}

TEST_CASE("Llama 3.2 encodes every fixture text to the references' token IDs") {
    check_cases(load("llama-3.2-1b-instruct-q4_0", kLlamaBpe), kLlamaCases);
}

TEST_CASE("encoding appends, and text that is not UTF-8 leaves the output untouched") {
    const auto bpe = load("qwen3-0.6b-q4_0", kQwen2);
    std::vector<TokenId> out{static_cast<TokenId>(7)};
    REQUIRE(bpe.encode("Hello", out) == EncodeError::Ok);
    CHECK(out.size() == 2);
    CHECK(out[0] == static_cast<TokenId>(7));
    CHECK(bpe.encode("ok \xC0\x80", out) == EncodeError::InvalidUtf8);
    CHECK(out.size() == 2);
}

TEST_CASE("a cache another tokenizer filled is emptied, never trusted") {
    const auto qwen = load("qwen3-0.6b-q4_0", kQwen2);
    const auto llama = load("llama-3.2-1b-instruct-q4_0", kLlamaBpe);
    bpe::PieceCache cache;
    for (const EncodeCase& c : kQwen3Cases) (void)encode(qwen, c.text, cache);
    // The same words, now Llama's: every one must be Llama's own tokens.
    for (const EncodeCase& c : kLlamaCases) {
        CAPTURE(c.name);
        CHECK(encode(llama, c.text, cache) == c.ids);
    }
}

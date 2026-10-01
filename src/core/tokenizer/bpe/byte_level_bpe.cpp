#include "core/tokenizer/bpe/byte_level_bpe.h"

#include <cstdio>
#include <limits>
#include <string>
#include <utility>

#include "core/tokenizer/bpe/byte_map.h"
#include "core/tokenizer/bpe/merge.h"
#include "core/tokenizer/nfc.h"
#include "core/tokenizer/unicode.h"

namespace bllm::tokenizer::bpe {

EncodeError ByteLevelBpe::encode(std::string_view text, std::vector<TokenId>& out) const {
    return encode_into(text, out, nullptr);
}

EncodeError ByteLevelBpe::encode(std::string_view text, std::vector<TokenId>& out, PieceCache& cache) const {
    if (cache.owner_ != this) {
        cache.clear();
        cache.owner_ = this;
    }
    return encode_into(text, out, &cache);
}

EncodeError ByteLevelBpe::encode_into(std::string_view text, std::vector<TokenId>& out, PieceCache* cache) const {
    if (text.size() > std::numeric_limits<std::uint32_t>::max()) return EncodeError::TooLong;

    std::vector<Segment> segments;
    special_.segment(text, segments);

    std::vector<TokenId> tokens;
    std::string normalized;
    std::vector<Piece> pieces;
    std::string spelled;              // scratch for each piece, kept across pieces
    std::vector<TokenId> symbols;
    for (const Segment& segment : segments) {
        if (segment.special) {
            tokens.push_back(*segment.special);
            continue;
        }
        std::string_view ordinary = text.substr(segment.offset, segment.length);
        if (pretokenizer_->normalization == Normalization::Nfc) {
            if (!to_nfc(ordinary, normalized)) return EncodeError::InvalidUtf8;
            ordinary = normalized;
        }
        switch (split(*pretokenizer_, ordinary, pieces)) {
            case SplitError::Ok: break;
            case SplitError::InvalidUtf8: return EncodeError::InvalidUtf8;
            case SplitError::TooLong: return EncodeError::TooLong;
        }
        for (const Piece& piece : pieces) {
            encode_piece(ordinary.substr(piece.offset, piece.length), tokens, cache, spelled, symbols);
        }
    }
    out.insert(out.end(), tokens.begin(), tokens.end());
    return EncodeError::Ok;
}

void ByteLevelBpe::encode_piece(std::string_view piece, std::vector<TokenId>& out, PieceCache* cache,
                                std::string& spelled, std::vector<TokenId>& symbols) const {
    if (cache != nullptr) {
        if (const auto hit = cache->find(piece)) {
            out.insert(out.end(), hit->begin(), hit->end());
            return;
        }
    }
    const std::size_t first = out.size();
    if (pretokenizer_->ignore_merges) {
        spelled.clear();
        for (const char byte : piece) append_utf8(kByteChars[static_cast<unsigned char>(byte)], spelled);
        if (const auto whole = vocabulary_.find(spelled)) out.push_back(*whole);
    }
    if (out.size() == first) {
        symbols.clear();
        for (const char byte : piece) symbols.push_back(byte_tokens_[static_cast<unsigned char>(byte)]);
        merge(symbols, merges_, out);
    }
    if (cache != nullptr) cache->insert(piece, std::span<const TokenId>{out}.subspan(first));
}

LoadResult load_byte_level_bpe(gguf::ByteSource& source, const gguf::TensorIndex& index,
                               const PreTokenizer& pretokenizer, ByteLevelBpe& out) {
    ByteLevelBpe bpe;
    if (auto r = load_vocabulary(source, index, bpe.vocabulary_); !r.ok()) return r;
    if (auto r = load_merges(source, index, bpe.vocabulary_, bpe.merges_); !r.ok()) return r;

    std::string spelled;
    for (std::size_t byte = 0; byte < 256; ++byte) {
        spelled.clear();
        append_utf8(kByteChars[byte], spelled);
        const auto token = bpe.vocabulary_.find(spelled);
        if (!token) {
            char name[8];
            std::snprintf(name, sizeof name, "0x%02zX", byte);
            return {LoadError::MissingByte, "byte " + std::string(name)};
        }
        bpe.byte_tokens_[byte] = *token;
    }
    bpe.special_ = SpecialTokens{bpe.vocabulary_};
    bpe.pretokenizer_ = &pretokenizer;
    out = std::move(bpe);
    return {};
}

}  // namespace bllm::tokenizer::bpe

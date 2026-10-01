#include "core/tokenizer/bpe/piece_cache.h"

#include <algorithm>
#include <bit>

namespace bllm::tokenizer::bpe {
namespace {

// FNV-1a, 64-bit: a short, well-spread hash for keys of a few bytes.
constexpr std::uint64_t kFnvOffset = 0xCBF29CE484222325;
constexpr std::uint64_t kFnvPrime = 0x100000001B3;

}  // namespace

PieceCache::PieceCache(std::size_t slots) : slots_(std::bit_ceil(std::max<std::size_t>(slots, 1)), Slot{}) {}

std::size_t PieceCache::slot_of(std::string_view piece) const noexcept {
    std::uint64_t hash = kFnvOffset;
    for (const char c : piece) hash = (hash ^ static_cast<unsigned char>(c)) * kFnvPrime;
    return static_cast<std::size_t>(hash) & (slots_.size() - 1);
}

std::optional<std::span<const TokenId>> PieceCache::find(std::string_view piece) const noexcept {
    if (piece.empty() || piece.size() > kMaxBytes) return std::nullopt;
    const Slot& slot = slots_[slot_of(piece)];
    if (slot.length != piece.size() || !std::equal(piece.begin(), piece.end(), slot.bytes.begin())) {
        return std::nullopt;
    }
    return std::span<const TokenId>{slot.tokens.data(), slot.count};
}

void PieceCache::insert(std::string_view piece, std::span<const TokenId> tokens) noexcept {
    if (piece.empty() || piece.size() > kMaxBytes || tokens.size() > kMaxTokens) return;
    Slot& slot = slots_[slot_of(piece)];
    std::copy(piece.begin(), piece.end(), slot.bytes.begin());
    std::copy(tokens.begin(), tokens.end(), slot.tokens.begin());
    slot.length = static_cast<std::uint8_t>(piece.size());
    slot.count = static_cast<std::uint8_t>(tokens.size());
}

void PieceCache::clear() noexcept {
    std::fill(slots_.begin(), slots_.end(), Slot{});
    owner_ = nullptr;
}

}  // namespace bllm::tokenizer::bpe

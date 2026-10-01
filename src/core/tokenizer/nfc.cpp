#include "core/tokenizer/nfc.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "core/tokenizer/unicode.h"
#include "core/tokenizer/unicode_tables.h"

namespace bllm::tokenizer {
namespace {

namespace tables = unicode_tables;

// Hangul syllables decompose and compose by arithmetic (Unicode Standard
// 3.12): a leading consonant, a vowel, and an optional trailing consonant.
constexpr char32_t kSyllableBase = 0xAC00;
constexpr char32_t kLeadBase = 0x1100;
constexpr char32_t kVowelBase = 0x1161;
constexpr char32_t kTrailBase = 0x11A7;   // one before the first trailing consonant
constexpr char32_t kLeadCount = 19;
constexpr char32_t kVowelCount = 21;
constexpr char32_t kTrailCount = 28;      // including "no trailing consonant"
constexpr char32_t kSyllableCount = kLeadCount * kVowelCount * kTrailCount;

// The lowest code point NFC can change or that can change a neighbour: below
// it every character's NFC quick check is Yes and its combining class is 0
// (DerivedNormalizationProps.txt and UnicodeData.txt, Unicode 16.0). Text of
// such characters alone is already in NFC, as most prompts are.
// Optimization (practice): the quick check of UAX #15, which normalizers such
// as ICU's apply before normalizing; it took NFC on 128 KiB of English from
// 8.8 ms to 0.1 ms.
constexpr char32_t kFirstChangeable = 0x0300;

std::uint8_t combining_class(char32_t c) {
    const auto table = tables::combining();
    const auto range = std::lower_bound(table.begin(), table.end(), c,
                                        [](const tables::CombiningRange& r, char32_t v) { return r.last < v; });
    return range != table.end() && range->first <= c ? range->combining_class : 0;
}

void decompose(char32_t c, std::vector<char32_t>& out) {
    if (c >= kSyllableBase && c < kSyllableBase + kSyllableCount) {
        const char32_t index = c - kSyllableBase;
        out.push_back(kLeadBase + index / (kVowelCount * kTrailCount));
        out.push_back(kVowelBase + index % (kVowelCount * kTrailCount) / kTrailCount);
        if (const char32_t trail = index % kTrailCount; trail != 0) out.push_back(kTrailBase + trail);
        return;
    }
    const auto table = tables::decompositions();
    const auto entry = std::lower_bound(table.begin(), table.end(), c,
                                        [](const tables::Decomposition& d, char32_t v) { return d.code_point < v; });
    if (entry == table.end() || entry->code_point != c) {
        out.push_back(c);
        return;
    }
    const auto pool = tables::decomposition_pool().subspan(entry->start, entry->length);
    out.insert(out.end(), pool.begin(), pool.end());
}

std::optional<char32_t> compose(char32_t first, char32_t second) {
    if (first >= kLeadBase && first < kLeadBase + kLeadCount &&
        second >= kVowelBase && second < kVowelBase + kVowelCount) {
        return kSyllableBase + ((first - kLeadBase) * kVowelCount + (second - kVowelBase)) * kTrailCount;
    }
    if (first >= kSyllableBase && first < kSyllableBase + kSyllableCount &&
        (first - kSyllableBase) % kTrailCount == 0 && second > kTrailBase && second < kTrailBase + kTrailCount) {
        return first + (second - kTrailBase);
    }
    const auto table = tables::compositions();
    const auto entry = std::lower_bound(table.begin(), table.end(), std::pair{first, second},
                                        [](const tables::Composition& c, const std::pair<char32_t, char32_t>& v) {
                                            return std::pair{c.first, c.second} < v;
                                        });
    if (entry != table.end() && entry->first == first && entry->second == second) return entry->composite;
    return std::nullopt;
}

// Sorts each run of combining marks by combining class. The sort is stable:
// marks of equal class keep their order, which is part of what they mean.
void put_in_canonical_order(std::vector<char32_t>& text) {
    const auto mark = [](char32_t c) { return combining_class(c) != 0; };
    for (auto run = text.begin(); run != text.end();) {
        run = std::find_if(run, text.end(), mark);
        const auto end = std::find_if_not(run, text.end(), mark);
        std::stable_sort(run, end, [](char32_t a, char32_t b) { return combining_class(a) < combining_class(b); });
        run = end;
    }
}

// Composes each character with the last starter before it, unless it is
// blocked: some character between them is a starter, or has a combining class
// at least its own. After canonical ordering the last character kept carries
// the highest class between, so it alone decides.
std::vector<char32_t> compose_all(const std::vector<char32_t>& text) {
    std::vector<char32_t> out;
    out.reserve(text.size());
    std::optional<std::size_t> starter;
    std::uint8_t previous = 0;   // the class of the last character kept
    for (const char32_t c : text) {
        const std::uint8_t cc = combining_class(c);
        if (starter) {
            const bool adjacent = *starter == out.size() - 1;
            if (adjacent || previous < cc) {
                if (const auto composite = compose(out[*starter], c)) {
                    out[*starter] = *composite;
                    continue;
                }
            }
        }
        if (cc == 0) starter = out.size();
        out.push_back(c);
        previous = cc;
    }
    return out;
}

}  // namespace

bool to_nfc(std::string_view text, std::string& out) {
    // Read until a character NFC could change. If there is none, the text is
    // its own NFC; it was still decoded throughout, so it is well formed.
    std::size_t at = 0;
    for (Utf8Char c{}; at < text.size(); at += c.length) {
        if (!decode_utf8(text, at, c)) return false;
        if (c.code_point >= kFirstChangeable) break;
    }
    if (at == text.size()) {
        out.assign(text);
        return true;
    }

    std::vector<char32_t> decomposed;
    decomposed.reserve(text.size());
    for (std::size_t at = 0; at < text.size();) {
        Utf8Char c{};
        if (!decode_utf8(text, at, c)) return false;
        decompose(c.code_point, decomposed);
        at += c.length;
    }
    put_in_canonical_order(decomposed);

    std::string normalized;
    normalized.reserve(text.size());
    for (const char32_t c : compose_all(decomposed)) append_utf8(c, normalized);
    out = std::move(normalized);
    return true;
}

}  // namespace bllm::tokenizer

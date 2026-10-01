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
// Optimization (practice): the quick check of UAX #15, applied span by span,
// as normalizers such as ICU's do: only the stretches around characters NFC
// could change are normalized. On this repository's docs and sources, 385 KB
// with 346 characters past ASCII scattered through, Qwen3's whole encode fell
// from 43 ms to 15 ms.
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

// Appends the NFC of `span`, which the caller has checked is well-formed
// UTF-8. `scratch` is reused from span to span.
void normalize_span(std::string_view span, std::vector<char32_t>& scratch, std::string& out) {
    scratch.clear();
    for (std::size_t at = 0; at < span.size();) {
        Utf8Char c{};
        (void)decode_utf8(span, at, c);
        decompose(c.code_point, scratch);
        at += c.length;
    }
    put_in_canonical_order(scratch);
    for (const char32_t c : compose_all(scratch)) append_utf8(c, out);
}

}  // namespace

bool to_nfc(std::string_view text, std::string& out) {
    // The boundary before a character below U+0300 is stable: that character
    // composes with nothing before it, and as a starter it blocks everything
    // after it from composing with anything before. So only a span from the
    // last such character before one that NFC could change, through the end of
    // the changeable run, is normalized; the rest is copied as it is. Every
    // byte is still decoded, so ill-formed UTF-8 is refused anywhere.
    std::string normalized;
    normalized.reserve(text.size());
    std::vector<char32_t> scratch;
    std::size_t copied = 0;   // text before this is in `normalized`
    std::size_t stable = 0;   // where the latest character below U+0300 starts
    for (std::size_t at = 0; at < text.size();) {
        Utf8Char c{};
        if (!decode_utf8(text, at, c)) return false;
        if (c.code_point < kFirstChangeable) {
            stable = at;
            at += c.length;
            continue;
        }
        std::size_t end = at + c.length;
        for (Utf8Char next{}; end < text.size(); end += next.length) {
            if (!decode_utf8(text, end, next)) return false;
            if (next.code_point < kFirstChangeable) break;
        }
        normalized.append(text, copied, stable - copied);
        normalize_span(text.substr(stable, end - stable), scratch, normalized);
        copied = stable = at = end;
    }
    normalized.append(text, copied, text.size() - copied);
    out = std::move(normalized);
    return true;
}

}  // namespace bllm::tokenizer

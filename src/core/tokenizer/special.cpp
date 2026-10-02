#include "core/tokenizer/special.h"

#include <algorithm>
#include <utility>

namespace bllm::tokenizer {

namespace {

std::vector<SpecialTokens::Entry> special_entries(const Vocabulary& vocabulary) {
    std::vector<SpecialTokens::Entry> entries;
    for (std::size_t i = 0; i < vocabulary.size(); ++i) {
        const auto id = static_cast<TokenId>(i);
        const TokenType type = vocabulary.type(id);
        if ((type == TokenType::Control || type == TokenType::UserDefined) && !vocabulary.text(id).empty()) {
            entries.push_back({std::string(vocabulary.text(id)), id});
        }
    }
    return entries;
}

}  // namespace

SpecialTokens::SpecialTokens(const Vocabulary& vocabulary) : SpecialTokens(special_entries(vocabulary)) {}

SpecialTokens::SpecialTokens(std::vector<Entry> entries) : tokens_(std::move(entries)) {
    const auto first_byte = [](const Entry& t) { return static_cast<unsigned char>(t.text[0]); };
    std::sort(tokens_.begin(), tokens_.end(), [&](const Entry& a, const Entry& b) {
        if (first_byte(a) != first_byte(b)) return first_byte(a) < first_byte(b);
        return a.text.size() > b.text.size();
    });
    std::size_t at = 0;
    for (std::size_t byte = 0; byte <= 256; ++byte) {
        while (at < tokens_.size() && first_byte(tokens_[at]) < byte) ++at;
        starts_[byte] = static_cast<std::uint32_t>(at);
    }
}

void SpecialTokens::segment(std::string_view text, std::vector<Segment>& out) const {
    std::size_t plain = 0;   // where the current run of ordinary text began
    for (std::size_t i = 0; i < text.size();) {
        const auto byte = static_cast<unsigned char>(text[i]);
        const Entry* match = nullptr;
        for (std::uint32_t t = starts_[byte]; t < starts_[byte + 1]; ++t) {
            if (text.substr(i).starts_with(tokens_[t].text)) {
                match = &tokens_[t];
                break;
            }
        }
        if (match == nullptr) {
            ++i;
            continue;
        }
        if (i > plain) {
            out.push_back({static_cast<std::uint32_t>(plain), static_cast<std::uint32_t>(i - plain), std::nullopt});
        }
        out.push_back({static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(match->text.size()), match->id});
        i += match->text.size();
        plain = i;
    }
    if (text.size() > plain) {
        out.push_back({static_cast<std::uint32_t>(plain), static_cast<std::uint32_t>(text.size() - plain),
                       std::nullopt});
    }
}

}  // namespace bllm::tokenizer

#include "core/gguf/reader.h"

#include <cstring>

#include "core/gguf/checked.h"
#include "core/quant/q4_0.h"

namespace bllm::gguf {
namespace {

// Little-endian decode. GGUF is little-endian on disk; doing this by hand
// rather than memcpy-ing into an integer keeps the code correct on a
// big-endian host without a conditional.
[[nodiscard]] std::uint32_t le32(const std::byte* p) noexcept {
    return static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(p[0])) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(p[1])) << 8) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(p[2])) << 16) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(p[3])) << 24);
}

[[nodiscard]] std::uint64_t le64(const std::byte* p) noexcept {
    return static_cast<std::uint64_t>(le32(p)) |
           (static_cast<std::uint64_t>(le32(p + 4)) << 32);
}

// Fixed sizes for the scalar metadata types.
[[nodiscard]] bool scalar_size(ValueType t, std::uint64_t& out) noexcept {
    switch (t) {
        case ValueType::UInt8: case ValueType::Int8: case ValueType::Bool:
            out = 1; return true;
        case ValueType::UInt16: case ValueType::Int16:
            out = 2; return true;
        case ValueType::UInt32: case ValueType::Int32: case ValueType::Float32:
            out = 4; return true;
        case ValueType::UInt64: case ValueType::Int64: case ValueType::Float64:
            out = 8; return true;
        case ValueType::String: case ValueType::Array:
            return false;   // variable length; handled separately
    }
    return false;
}

[[nodiscard]] bool known_value_type(std::uint32_t raw) noexcept {
    return raw <= static_cast<std::uint32_t>(ValueType::Float64);
}

}  // namespace

bool bytes_for_elements(TensorType type, std::uint64_t elements,
                        std::uint64_t& out) noexcept {
    switch (type) {
        case TensorType::F32:
            return checked_mul(elements, 4, out);
        case TensorType::F16:
            return checked_mul(elements, 2, out);
        case TensorType::Q4_0: {
            if (!quant::is_block_aligned(static_cast<std::size_t>(elements))) {
                return false;
            }
            std::uint64_t blocks = elements / quant::kQ4_0BlockElements;
            return checked_mul(blocks, quant::kQ4_0BlockBytes, out);
        }
        default:
            return false;   // valid ggml type, not one this harness reads
    }
}

ReadError Reader::take(std::uint64_t count, std::span<std::byte> out) {
    if (!range_within(cursor_, count, source_.size())) {
        return ReadError::ShortRead;
    }
    if (!source_.read(cursor_, out.first(static_cast<std::size_t>(count)))) {
        return ReadError::ShortRead;
    }
    cursor_ += count;
    return ReadError::Ok;
}

ReadError Reader::take_u32(std::uint32_t& out) {
    std::byte buf[4];
    if (const auto e = take(4, buf); e != ReadError::Ok) return e;
    out = le32(buf);
    return ReadError::Ok;
}

ReadError Reader::take_u64(std::uint64_t& out) {
    std::byte buf[8];
    if (const auto e = take(8, buf); e != ReadError::Ok) return e;
    out = le64(buf);
    return ReadError::Ok;
}

ReadError Reader::take_string(std::string& out) {
    std::uint64_t length = 0;
    if (const auto e = take_u64(length); e != ReadError::Ok) return e;
    // Bounded before allocating: a hostile length must not be an allocation
    // request.
    if (length > kMaxStringLength) return ReadError::StringTooLong;
    if (!range_within(cursor_, length, source_.size())) return ReadError::ShortRead;

    out.resize(static_cast<std::size_t>(length));
    if (length != 0) {
        std::span<std::byte> dest{reinterpret_cast<std::byte*>(out.data()),
                                  static_cast<std::size_t>(length)};
        if (!source_.read(cursor_, dest)) return ReadError::ShortRead;
    }
    cursor_ += length;
    return ReadError::Ok;
}

ReadError Reader::skip_value(ValueType type, std::uint64_t& consumed) {
    const std::uint64_t start = cursor_;

    if (type == ValueType::String) {
        std::string ignored;
        if (const auto e = take_string(ignored); e != ReadError::Ok) return e;
    } else if (type == ValueType::Array) {
        std::uint32_t raw_element_type = 0;
        if (const auto e = take_u32(raw_element_type); e != ReadError::Ok) return e;
        if (!known_value_type(raw_element_type)) return ReadError::UnknownValueType;
        const auto element_type = static_cast<ValueType>(raw_element_type);
        if (element_type == ValueType::Array) return ReadError::NestedArray;

        std::uint64_t count = 0;
        if (const auto e = take_u64(count); e != ReadError::Ok) return e;
        if (count > kMaxArrayLength) return ReadError::ArrayTooLong;

        if (element_type == ValueType::String) {
            // Variable-length elements must be walked; there is no shortcut,
            // but nothing is materialised.
            for (std::uint64_t i = 0; i < count; ++i) {
                std::string ignored;
                if (const auto e = take_string(ignored); e != ReadError::Ok) return e;
            }
        } else {
            std::uint64_t element_size = 0;
            if (!scalar_size(element_type, element_size)) return ReadError::UnknownValueType;
            std::uint64_t total = 0;
            if (!checked_mul(count, element_size, total)) return ReadError::OffsetOverflow;
            if (!range_within(cursor_, total, source_.size())) return ReadError::ShortRead;
            cursor_ += total;
        }
    } else {
        std::uint64_t size = 0;
        if (!scalar_size(type, size)) return ReadError::UnknownValueType;
        if (!range_within(cursor_, size, source_.size())) return ReadError::ShortRead;
        cursor_ += size;
    }

    consumed = cursor_ - start;
    return ReadError::Ok;
}

const MetadataEntry* Reader::find(std::string_view key) const noexcept {
    for (const auto& e : metadata_) {
        if (e.key == key) return &e;
    }
    return nullptr;
}

ReadError Reader::parse() {
    cursor_ = 0;

    std::byte magic[4];
    if (const auto e = take(4, magic); e != ReadError::Ok) return e;
    for (int i = 0; i < 4; ++i) {
        if (std::to_integer<char>(magic[i]) != kMagic[i]) return ReadError::BadMagic;
    }

    std::uint32_t version = 0;
    if (const auto e = take_u32(version); e != ReadError::Ok) return e;
    if (version != kSupportedVersion) return ReadError::UnsupportedVersion;

    std::uint64_t tensor_count = 0;
    std::uint64_t metadata_count = 0;
    if (const auto e = take_u64(tensor_count); e != ReadError::Ok) return e;
    if (const auto e = take_u64(metadata_count); e != ReadError::Ok) return e;
    // Bounded before reserving: a declared count is a claim, not a fact.
    if (tensor_count > kMaxTensorCount) return ReadError::CountTooLarge;
    if (metadata_count > kMaxMetadataCount) return ReadError::CountTooLarge;

    metadata_.clear();
    metadata_.reserve(static_cast<std::size_t>(metadata_count));
    for (std::uint64_t i = 0; i < metadata_count; ++i) {
        MetadataEntry entry;
        if (const auto e = take_string(entry.key); e != ReadError::Ok) return e;

        std::uint32_t raw_type = 0;
        if (const auto e = take_u32(raw_type); e != ReadError::Ok) return e;
        if (!known_value_type(raw_type)) return ReadError::UnknownValueType;
        entry.type = static_cast<ValueType>(raw_type);

        entry.value_offset = cursor_;
        if (const auto e = skip_value(entry.type, entry.value_length); e != ReadError::Ok) {
            return e;
        }
        metadata_.push_back(std::move(entry));
    }

    // general.alignment, if present, governs where tensor data begins.
    alignment_ = kDefaultAlignment;
    if (const MetadataEntry* a = find("general.alignment");
        a != nullptr && a->type == ValueType::UInt32) {
        std::byte buf[4];
        if (source_.read(a->value_offset, buf)) {
            const std::uint32_t value = le32(buf);
            // Must be a power of two: a zero or non-power-of-two alignment
            // would make the padding computation meaningless.
            if (value != 0 && (value & (value - 1)) == 0) {
                alignment_ = value;
            }
        }
    }

    tensors_.clear();
    tensors_.reserve(static_cast<std::size_t>(tensor_count));
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        TensorEntry t;
        if (const auto e = take_string(t.name); e != ReadError::Ok) return e;

        for (const auto& existing : tensors_) {
            if (existing.name == t.name) return ReadError::DuplicateTensorName;
        }

        if (const auto e = take_u32(t.dimension_count); e != ReadError::Ok) return e;
        if (t.dimension_count > kMaxDimensions) return ReadError::TooManyDimensions;

        t.element_count = 1;
        for (std::uint32_t d = 0; d < t.dimension_count; ++d) {
            std::uint64_t ne = 0;
            if (const auto e = take_u64(ne); e != ReadError::Ok) return e;
            // Stored as int64 on disk; a negative value arrives with the top
            // bit set and would otherwise become an enormous positive extent.
            if ((ne >> 63) != 0) return ReadError::NegativeDimension;
            t.dimensions[d] = ne;
            if (!checked_mul(t.element_count, ne, t.element_count)) {
                return ReadError::ElementCountOverflow;
            }
        }

        std::uint32_t raw_tensor_type = 0;
        if (const auto e = take_u32(raw_tensor_type); e != ReadError::Ok) return e;
        if (raw_tensor_type >= static_cast<std::uint32_t>(TensorType::Count)) {
            return ReadError::UnknownTensorType;
        }
        t.type = static_cast<TensorType>(raw_tensor_type);

        std::uint64_t relative_offset = 0;
        if (const auto e = take_u64(relative_offset); e != ReadError::Ok) return e;
        t.data_offset = relative_offset;   // made absolute below

        if (!bytes_for_elements(t.type, t.element_count, t.data_length)) {
            return ReadError::UnsupportedTensorType;
        }
        tensors_.push_back(std::move(t));
    }

    // Tensor data begins at the next alignment boundary after the index.
    const std::uint64_t pad = (alignment_ - (cursor_ % alignment_)) % alignment_;
    if (!checked_add(cursor_, pad, data_start_)) return ReadError::OffsetOverflow;
    if (data_start_ > source_.size()) return ReadError::TensorDataOutOfBounds;

    // Tensor offsets are relative to data_start_. Make them absolute and
    // validate each region against the file's authoritative size.
    for (auto& t : tensors_) {
        std::uint64_t absolute = 0;
        if (!checked_add(data_start_, t.data_offset, absolute)) {
            return ReadError::OffsetOverflow;
        }
        if (!range_within(absolute, t.data_length, source_.size())) {
            return ReadError::TensorDataOutOfBounds;
        }
        t.data_offset = absolute;
    }

    return ReadError::Ok;
}

}  // namespace bllm::gguf

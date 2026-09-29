#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "core/gguf/types.h"

namespace bllm::gguf {

// Contract 2: the tensor index.
//
// What the reader produces and every later stage reads. The reader is a
// parser; this is its result, and the reader does not outlive the parse.
//
//   - Immutable. Built once by the reader, read-only afterwards, and alive for
//     as long as anything refers to a tensor by identifier.
//   - Complete. Every tensor in the file, including formats the harness cannot
//     run. Whether a format runs is the capability table's question; the index
//     records what the file says.
//   - Scalar and string metadata is decoded during the parse. A missing key
//     and a key of the wrong type are different errors because they are
//     different defects: one is a file that lacks a field, the other a file
//     that misdeclares it.
//   - Arrays are located, not decoded. A vocabulary is ~150,000 strings and
//     only the tokenizer reads it, from the byte source.

// A tensor's position in the index. Its own type, so a tensor identifier
// cannot be passed where a layer or a token is meant (I.4).
enum class TensorId : std::uint32_t {};

enum class MetadataError {
    Ok,
    MissingKey,
    WrongType,
};

// Where an array value lives in the file.
struct ArrayLocation {
    ValueType element_type;
    std::uint64_t element_count;
    ByteRange bytes;
};

class TensorIndex {
public:
    [[nodiscard]] std::span<const TensorEntry> tensors() const noexcept;

    // Precondition: `id` came from this index.
    [[nodiscard]] const TensorEntry& tensor(TensorId id) const noexcept;

    [[nodiscard]] std::optional<TensorId> find(std::string_view name) const noexcept;

    [[nodiscard]] MetadataError read_u32(std::string_view key, std::uint32_t& out) const noexcept;
    [[nodiscard]] MetadataError read_u64(std::string_view key, std::uint64_t& out) const noexcept;
    [[nodiscard]] MetadataError read_f32(std::string_view key, float& out) const noexcept;
    [[nodiscard]] MetadataError read_bool(std::string_view key, bool& out) const noexcept;
    // `out` views storage owned by the index and lives as long as it does.
    [[nodiscard]] MetadataError read_string(std::string_view key, std::string_view& out) const noexcept;
    [[nodiscard]] MetadataError read_array(std::string_view key, ArrayLocation& out) const noexcept;

private:
    // Built only by the reader.
    friend class Reader;
    TensorIndex() = default;

    std::vector<TensorEntry> tensors_;
};

}  // namespace bllm::gguf

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf/byte_source.h"
#include "core/gguf/types.h"

namespace bllm::gguf {

// Where a metadata value lives, rather than the value itself.
//
// Recording the location and decoding on demand keeps peak memory bounded: the
// tokenizer vocabulary is a ~152,000-entry string array, and materialising it
// while merely locating the tensor index would defeat the streaming contract.
struct MetadataEntry {
    std::string key;
    ValueType type = ValueType::UInt32;
    std::uint64_t value_offset = 0;   // absolute, start of the encoded value
    std::uint64_t value_length = 0;   // bytes the encoded value occupies
};

// Parses a GGUF header, metadata index and tensor index from an untrusted
// source.
//
// Contract:
//   - Never reads outside the file. Every offset and length is validated
//     against the source's authoritative size, with checked arithmetic.
//   - Never throws. Failures are named values (E.27).
//   - Reads only the header, metadata locations and tensor index — never
//     tensor payloads. The whole file is not resident at any point.
class Reader {
public:
    explicit Reader(ByteSource& source) noexcept : source_(source) {}

    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    // Parses the file's index structures. On failure the reader's contents are
    // unspecified and must not be used.
    [[nodiscard]] ReadError parse();

    [[nodiscard]] const std::vector<TensorEntry>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] const std::vector<MetadataEntry>& metadata() const noexcept { return metadata_; }

    // Alignment declared by general.alignment, or the format default.
    [[nodiscard]] std::uint64_t alignment() const noexcept { return alignment_; }
    // First byte of the tensor data region.
    [[nodiscard]] std::uint64_t tensor_data_start() const noexcept { return data_start_; }

    [[nodiscard]] const MetadataEntry* find(std::string_view key) const noexcept;

private:
    // Cursor-based reads; each advances `cursor_` only on success.
    [[nodiscard]] ReadError take(std::uint64_t count, std::span<std::byte> out);
    [[nodiscard]] ReadError take_u32(std::uint32_t& out);
    [[nodiscard]] ReadError take_u64(std::uint64_t& out);
    [[nodiscard]] ReadError take_string(std::string& out);
    // Advances past a value of the given type without materialising it.
    [[nodiscard]] ReadError skip_value(ValueType type, std::uint64_t& consumed);

    ByteSource& source_;
    std::uint64_t cursor_ = 0;
    std::uint64_t alignment_ = kDefaultAlignment;
    std::uint64_t data_start_ = 0;
    std::vector<TensorEntry> tensors_;
    std::vector<MetadataEntry> metadata_;
};

// Bytes occupied by `elements` values of `type`. False if the type is not one
// this harness handles, or if the product overflows.
[[nodiscard]] bool bytes_for_elements(TensorType type, std::uint64_t elements,
                                      std::uint64_t& out) noexcept;

}  // namespace bllm::gguf

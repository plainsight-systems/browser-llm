#pragma once

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <string_view>
#include <vector>

namespace bllm::gguf {

inline constexpr char kMagic[4] = {'G', 'G', 'U', 'F'};
inline constexpr std::uint32_t kSupportedVersion = 3;
inline constexpr std::uint32_t kDefaultAlignment = 32;
inline constexpr std::uint32_t kMaxDimensions = 4;

// Bounds that exist to stop a hostile file from making us allocate. They are
// not format limits, they are our limits, and exceeding one is a named error
// rather than an attempt that happens to fail.
inline constexpr std::uint64_t kMaxTensorCount = 1u << 20;
inline constexpr std::uint64_t kMaxMetadataCount = 1u << 20;
inline constexpr std::uint64_t kMaxStringLength = 1u << 20;
inline constexpr std::uint64_t kMaxArrayLength = 1u << 24;

// GGUF metadata value types, matching the format's own numbering.
enum class ValueType : std::uint32_t {
    UInt8 = 0, Int8 = 1, UInt16 = 2, Int16 = 3,
    UInt32 = 4, Int32 = 5, Float32 = 6, Bool = 7,
    String = 8, Array = 9, UInt64 = 10, Int64 = 11, Float64 = 12,
};

// ggml tensor types. Only the ones this project reads are named; the rest are
// recognised as valid-but-unsupported so an unexpected file fails with a
// useful error rather than "unknown type".
enum class TensorType : std::uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3,
    Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q6_K = 14, Q8_K = 15,
    Count = 40,   // upper bound for range validation
};

// The closed set of ways reading can fail. One vocabulary for callers (E.27);
// the reader never throws and never reads out of bounds.
enum class ReadError : std::uint32_t {
    Ok = 0,
    ShortRead,               // the source could not supply the bytes
    BadMagic,
    UnsupportedVersion,
    CountTooLarge,           // tensor or metadata count beyond our bound
    StringTooLong,
    ArrayTooLong,
    UnknownValueType,
    UnknownTensorType,
    UnsupportedTensorType,   // valid ggml type, not one we handle
    TooManyDimensions,
    NegativeDimension,
    ElementCountOverflow,    // the shape product does not fit
    OffsetOverflow,          // offset + length wraps
    TensorDataOutOfBounds,   // the region is not inside the file
    DuplicateTensorName,
    MisalignedTensorData,
    NestedArray,             // arrays of arrays are not supported
};

[[nodiscard]] constexpr std::string_view to_string(ReadError e) noexcept {
    switch (e) {
        case ReadError::Ok: return "ok";
        case ReadError::ShortRead: return "short read: the file ends before this field";
        case ReadError::BadMagic: return "not a GGUF file: magic mismatch";
        case ReadError::UnsupportedVersion: return "unsupported GGUF version";
        case ReadError::CountTooLarge: return "declared count exceeds the reader's bound";
        case ReadError::StringTooLong: return "string length exceeds the reader's bound";
        case ReadError::ArrayTooLong: return "array length exceeds the reader's bound";
        case ReadError::UnknownValueType: return "unknown metadata value type";
        case ReadError::UnknownTensorType: return "tensor type outside the ggml range";
        case ReadError::UnsupportedTensorType: return "tensor type not supported by this harness";
        case ReadError::TooManyDimensions: return "tensor declares more than 4 dimensions";
        case ReadError::NegativeDimension: return "tensor declares a negative dimension";
        case ReadError::ElementCountOverflow: return "tensor element count overflows";
        case ReadError::OffsetOverflow: return "offset plus length overflows";
        case ReadError::TensorDataOutOfBounds: return "tensor data lies outside the file";
        case ReadError::DuplicateTensorName: return "duplicate tensor name";
        case ReadError::MisalignedTensorData: return "tensor data start is misaligned";
        case ReadError::NestedArray: return "nested arrays are not supported";
    }
    return "unrecognised error";
}

// A tensor's extent, grouped so it cannot be passed to a constructor in
// pieces or misordered against the byte range.
struct TensorShape {
    std::uint32_t dimension_count = 0;
    std::uint64_t dimensions[kMaxDimensions] = {1, 1, 1, 1};
    std::uint64_t element_count = 0;
};

// A byte region within the file.
struct ByteRange {
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
};

// One tensor's entry in the index. Holds offsets and lengths, never pointers,
// so it cannot outlive a buffer into undefined behaviour.
//
// Quantization travels WITH the tensor by construction: there is no way to
// obtain a weight reference without also obtaining what it takes to interpret
// it. A design where the loader "handles quantization" and consumers see plain
// bytes is the design that silently produces uninterpretable values.
struct TensorEntry {
    // There is no default constructor, and `type` has no default value.
    //
    // A default-constructed entry would be a zero-length F32 tensor: a record
    // that looks valid, reports is_quantized() == false, and is wrong. The
    // type must be supplied to build the object at all, rather than assigned
    // afterwards by a step somebody can forget (C.41, NR.5).
    //
    // The parameters are four mutually non-confusable types. A flat
    // constructor would take four interchangeable std::uint64_t values, which
    // trades one silent-misinitialization defect for a worse one.
    TensorEntry() = delete;

    TensorEntry(std::string tensor_name, TensorType tensor_type,
                const TensorShape& shape, const ByteRange& data)
        : name(std::move(tensor_name)),
          type(tensor_type),
          dimension_count(shape.dimension_count),
          dimensions{shape.dimensions[0], shape.dimensions[1],
                     shape.dimensions[2], shape.dimensions[3]},
          element_count(shape.element_count),
          data_offset(data.offset),
          data_length(data.length) {}

    std::string name;
    TensorType type;
    std::uint32_t dimension_count;
    std::uint64_t dimensions[kMaxDimensions];

    std::uint64_t element_count;
    // Byte range within the file. Relative to the tensor data region while the
    // index is being read; absolute — and validated as inside the file — once
    // Reader::parse returns Ok.
    std::uint64_t data_offset;
    std::uint64_t data_length;

    [[nodiscard]] bool is_quantized() const noexcept {
        return type != TensorType::F32 && type != TensorType::F16;
    }
};

// The invariant, asserted rather than described. If someone restores a default
// this fails at compile time, in this file, next to the reason.
static_assert(!std::is_default_constructible_v<TensorEntry>,
              "a TensorEntry without a type is the defect this type prevents");

}  // namespace bllm::gguf

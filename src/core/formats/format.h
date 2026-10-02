#pragma once

#include <string_view>

#include "core/formats/device_layout.h"
#include "core/gguf/types.h"

namespace bllm::formats {

// Contract: what every weight format supplies.
//
// Each formats/<format>/ provides one Format, and the capability table lists
// it under the GGUF type it implements. A format is the only code that knows
// its block layout's meaning. It supplies:
//
//   - unpack, as WGSL a kernel composes with. No kernel knows a format; adding
//     one adds a file here and a row in the capability table, and touches no
//     kernel.
//   - pack, as WGSL, for a format the KV cache stores in. Packing and
//     unpacking are one piece of knowledge, whether the data is a weight or a
//     cached key.
//   - its device layout: how its blocks' fields lie on the device, as streams
//     (device_layout.h). Upload writes a weight that way; unpack reads it.
//
// A Format cannot be built wrong. It exists only at compile time (each
// formats/<format>/ defines one as a constexpr constant), it takes its type
// from its layout, so the two cannot name different types, and it has no
// state without a layout. Its constructor refuses, at compile time, a layout
// that breaks the promise device_layout.h states, and an empty unpack: a
// format with nothing to read its weights is not a format this build runs. The capability table then
// checks at compile time that every row's type is its Format's type. So a
// format the table lists always has a layout that matches it and keeps its
// promise, and preflight and upload cannot disagree on how a format runs.
//
// Block sizes belong to the file format and are read from core/gguf; a format
// does not restate them. There is no CPU dequantizer: production never
// materialises a dequantized weight, and the CPU reference used to check
// unpack lives under tests/.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     P.5    Prefer compile-time checking to run-time checking — a wrong
//            pairing of type and layout does not compile.
//     C.41   A constructor should create a fully initialized object — there
//            is no Format without a layout.

namespace detail {
// Not constexpr: reached only when a Format would be built wrong, which makes
// its construction fail to compile.
void layout_breaks_its_promise();
void format_has_no_unpack();
}  // namespace detail

class Format {
public:
    // `pack_wgsl` is empty for a format the cache never stores in.
    consteval Format(const DeviceLayout& layout, std::string_view unpack_wgsl,
                     std::string_view pack_wgsl = {})
        : layout_(&layout), unpack_wgsl_(unpack_wgsl), pack_wgsl_(pack_wgsl) {
        if (!keeps_its_promise(layout)) detail::layout_breaks_its_promise();
        if (unpack_wgsl.empty()) detail::format_has_no_unpack();
    }

    [[nodiscard]] constexpr gguf::TensorType type() const noexcept { return layout_->type; }
    [[nodiscard]] constexpr const DeviceLayout& layout() const noexcept { return *layout_; }
    [[nodiscard]] constexpr std::string_view unpack_wgsl() const noexcept { return unpack_wgsl_; }
    [[nodiscard]] constexpr std::string_view pack_wgsl() const noexcept { return pack_wgsl_; }

private:
    const DeviceLayout* layout_;   // never null: the constructor takes a reference
    std::string_view unpack_wgsl_;
    std::string_view pack_wgsl_;
};

}  // namespace bllm::formats

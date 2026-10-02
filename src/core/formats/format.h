#pragma once

#include <string_view>

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
//   - a device layout: how its blocks' fields lie on the device, as streams
//     (device_layout.h). Upload writes a weight that way; unpack reads it.
//
// Block sizes belong to the file format and are read from core/gguf; a format
// does not restate them. There is no CPU dequantizer: production never
// materialises a dequantized weight, and the CPU reference used to check
// unpack lives under tests/.

struct Format {
    gguf::TensorType type;
    std::string_view unpack_wgsl;
    // Empty for a format the cache never stores in.
    std::string_view pack_wgsl;
};

}  // namespace bllm::formats

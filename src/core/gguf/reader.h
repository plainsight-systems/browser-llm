#pragma once

#include "core/gguf/byte_source.h"
#include "core/gguf/index.h"
#include "core/gguf/types.h"

namespace bllm::gguf {

// Axis C: changes with the GGUF format, or with this parse contract.
//
// Reads a GGUF file's header, metadata and tensor index from an untrusted
// source into `out`.
//
//   - Never reads outside the file. Every offset and length is validated
//     against the source's authoritative size, with checked arithmetic.
//   - Never throws. Failures are named values (E.27).
//   - Reads the index only, never tensor data, so only the front of the file
//     needs to be resident. When a read reaches bytes that are not,
//     the result is NeedMoreBytes with how many bytes to supply.
//   - Records every tensor, whatever its format. A format the harness cannot
//     run is not an error here; the gates decide, and name the tensor.
//   - Leaves `out` untouched unless the read succeeds.
[[nodiscard]] ReadResult read_index(ByteSource& source, TensorIndex& out);

}  // namespace bllm::gguf

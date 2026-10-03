#pragma once

#include "bllm/shaders_generated.h"
#include "core/formats/device_layout.h"
#include "core/formats/format.h"

namespace bllm::formats {

// Axis B: changes with a new weight format.
//
// Q6_K: 256 weights a super-block, six bits each, with sixteen 8-bit scales
// and an fp16 d, 6.5625 bits a weight; the listed models' token embeddings.
// Its unpack is q6_k.wgsl; its layout is kQ6_KLayout (device_layout.h), of
// the block as upload repacks it (q6_k_repack.cpp), 210 bytes as stored:
//   lo      bytes 0 .. 127: group k's low nibbles at 16k, byte j holding
//           weight j's in its low half and weight j + 16's in its high
//   hi      bytes 128 .. 191: group k's high bit pairs at 8k, byte m holding
//           weights m, m + 8, m + 16 and m + 24 at bits 0, 2, 4 and 6
//   scales  bytes 192 .. 207, as stored: group k's are 2k and 2k + 1
//   d       bytes 208 .. 209, as stored
inline constexpr Format kQ6_K{kQ6_KLayout, shaders::q6_k};

}  // namespace bllm::formats

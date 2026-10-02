#pragma once

#include "bllm/shaders_generated.h"
#include "core/formats/device_layout.h"
#include "core/formats/format.h"

namespace bllm::formats {

// Axis B: changes with a new weight format.
//
// Q6_K: 256 weights a super-block, six bits each, with sixteen 8-bit scales
// and an fp16 d, 6.5625 bits a weight; the listed models' token embeddings.
// Its unpack is q6_k.wgsl; its layout, ql, qh, scales then d, is kQ6_KLayout
// (device_layout.h).
inline constexpr Format kQ6_K{kQ6_KLayout, shaders::q6_k};

}  // namespace bllm::formats

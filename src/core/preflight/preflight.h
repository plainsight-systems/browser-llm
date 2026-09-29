#pragma once

#include <string>
#include <vector>

#include "core/gguf/index.h"
#include "core/policy/policy.h"
#include "core/residency/plan.h"

namespace bllm::preflight {

// Axis I: applies the capability table; it holds no criteria of its own.
//
// The four gates, run on the tensor index before any weight byte is fetched:
//
//   1. Architecture — the capability table has general.architecture, and that
//      architecture can describe this file.
//   2. Formats — the capability table has every tensor's format.
//   3. Tokenizer — the capability table has the algorithm, and the
//      pre-tokenizer when the algorithm needs one.
//   4. Device fit — the residency plan succeeds within the granted limits and
//      the memory budget in load policy.
//
// A rejection names what failed: the architecture, each unsupported format
// with the first tensor that uses it, the tokenizer or pre-tokenizer, or bytes
// needed against bytes available. Every gate that can run does, and the
// verdict lists every rejection rather than the first. Gate 4 needs the
// architecture's description, so it runs only when gate 1 passes.

enum class Gate {
    Architecture,
    Formats,
    Tokenizer,
    DeviceFit,
};

struct Rejection {
    Gate gate;
    std::string detail;
};

struct Verdict {
    std::vector<Rejection> rejections;

    [[nodiscard]] bool accepted() const noexcept { return rejections.empty(); }
};

[[nodiscard]] Verdict preflight(const gguf::TensorIndex& index,
                                const residency::DeviceLimits& limits,
                                const policy::LoadPolicy& policy);

}  // namespace bllm::preflight

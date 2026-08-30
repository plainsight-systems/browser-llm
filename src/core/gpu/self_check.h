#pragma once

#include <cstddef>
#include <string>

#include "core/gpu/device.h"

namespace bllm::gpu {

struct SelfCheckResult {
    bool ok = false;
    std::string error;       // populated only when ok is false
    std::size_t elements = 0;
    std::size_t mismatches = 0;
};

// Runs the embedded vector_add shader over `elements` values and verifies the
// readback against the values computed on the CPU.
//
// This proves the toolchain reaches the GPU and returns correct data. It is
// not a model kernel: correctness here is self-evident, which is why it needs
// no reference oracle. Readback is asynchronous, so the result is delivered by
// continuation rather than returned.
using SelfCheckCallback = void (*)(const SelfCheckResult& result, void* userdata);

void run_self_check(Device& device, std::size_t elements,
                    SelfCheckCallback callback, void* userdata);

}  // namespace bllm::gpu

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "core/gpu/device.h"

namespace bllm::gpu {

// Measures the cost of getting a value back from the GPU.
//
// This exists to answer one architectural question before the decode loop is
// designed: what does a serialized submit -> map -> read round trip cost? A
// decode loop that reads the sampled token back every step is bounded by that
// latency no matter how fast the kernels are, so the number sets a ceiling on
// tokens per second for that design.
//
// Two phases, because the comparison is the point:
//   sequential — N round trips, each waiting for the last. The per-token model.
//   batched    — N dispatches submitted, then one readback. The alternative.
//
// This is a measurement spike, not part of the inference path. It is compiled
// in deliberately and run only on request.
struct ReadbackBenchResult {
    bool ok = false;
    std::string error;

    std::size_t iterations = 0;
    std::vector<double> sequential_ms;   // one entry per round trip
    double batched_total_ms = 0.0;       // N dispatches, single readback

    std::unique_ptr<Device> device;      // returned, as run_self_check does
};

// Injected so core does not reach for a platform clock. Must return
// monotonically increasing milliseconds.
using NowFn = double (*)();

using BenchCallback = void (*)(ReadbackBenchResult result, void* userdata);

void run_readback_bench(std::unique_ptr<Device> device, std::size_t iterations,
                        NowFn now, BenchCallback callback, void* userdata);

}  // namespace bllm::gpu

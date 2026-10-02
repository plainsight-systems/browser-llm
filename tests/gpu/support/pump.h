#pragma once

#include <doctest/doctest.h>

#include <chrono>
#include <string>
#include <thread>

#include <webgpu/webgpu.h>

// Runs WebGPU callbacks natively — TEST SUPPORT ONLY. Natively every callback
// is AllowProcessEvents (core/gpu/callback_mode.h): it runs only inside
// wgpuInstanceProcessEvents, on the thread that calls it, as the browser's one
// event loop would run it. A wait that does not end fails the test, by name,
// rather than hanging it.
namespace bllm::testing {

inline void pump_until(WGPUInstance instance, const bool& done, const char* waiting_for,
                       std::chrono::seconds timeout = std::chrono::seconds{30}) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!done) {
        wgpuInstanceProcessEvents(instance);
        if (done) break;
        if (std::chrono::steady_clock::now() > deadline) FAIL("timed out waiting for " << std::string(waiting_for));
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
}

}  // namespace bllm::testing

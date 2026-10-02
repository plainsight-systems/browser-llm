#pragma once

#include <webgpu/webgpu.h>

namespace bllm::gpu {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// How every WebGPU callback this harness asks for is delivered. The harness is
// single-threaded (no pthreads: GitHub Pages cannot set COOP/COEP), and every
// callback must run on that one thread, between the harness's own steps.
//
//   - In the browser, AllowSpontaneous: emdawnwebgpu runs callbacks from the
//     worker's own event loop, the one thread there is, and nothing else
//     would ever run them.
//   - Natively, AllowProcessEvents: Dawn may run a spontaneous callback on a
//     thread of its own, which the harness's state is not built for. Here a
//     callback runs only inside wgpuInstanceProcessEvents, which the native
//     tests call from their one thread (tests/gpu/support/pump.h).
//
// The build chooses, not the core: CMakeLists.txt defines
// BLLM_GPU_PROCESS_EVENTS as 1 natively and 0 for the browser, and a build
// that defines neither does not compile, so none can fall into the wrong mode
// unnoticed. The only place the two builds differ in how the GPU talks back.
#if !defined(BLLM_GPU_PROCESS_EVENTS)
#error "BLLM_GPU_PROCESS_EVENTS is set by CMakeLists.txt: 1 natively, 0 in the browser"
#endif

inline constexpr WGPUCallbackMode kCallbackMode =
#if BLLM_GPU_PROCESS_EVENTS
    WGPUCallbackMode_AllowProcessEvents;
#else
    WGPUCallbackMode_AllowSpontaneous;
#endif

}  // namespace bllm::gpu

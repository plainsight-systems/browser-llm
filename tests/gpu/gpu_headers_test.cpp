// Compile check only: the core headers that hold WebGPU objects build natively,
// together, against Dawn, with nothing else included first. The platform-
// neutral ones are checked in tests/contract_headers_test.cpp.

#include "core/gpu/callback_mode.h"
#include "core/gpu/device.h"
#include "core/gpu/self_check.h"
#include "core/gpu/wgpu_handles.h"
#include "core/residency/upload.h"
#include "core/residency/upload_check.h"

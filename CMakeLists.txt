cmake_minimum_required(VERSION 3.25)
project(browser_llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(EmbedShaders)

# ---------------------------------------------------------------------------
# Core: platform-neutral C++. Compiles unchanged in both configurations.
#
# Invariant: nothing here may include <emscripten.h> or anything under
# src/wasm/ or web/. tools/check_boundaries.sh enforces this in CI.
# ---------------------------------------------------------------------------
# Diagnostic instrumentation is compiled out unless explicitly requested. See
# docs/research/2026-08-31-measurement-build-configurations.md — the axis is
# instrumentation, not optimisation, so a diagnostic build is still Release.
option(BLLM_ENABLE_DIAGNOSTICS
       "Compile diagnostic instrumentation into the build (perturbs timing)" OFF)

add_library(browser_llm_core STATIC
    src/core/gpu/dispatch_math.cpp
)

if(BLLM_ENABLE_DIAGNOSTICS)
    target_compile_definitions(browser_llm_core PUBLIC BLLM_DIAGNOSTICS_ENABLED=1)
    message(STATUS
        "browser-llm: diagnostics ENABLED — timings from this build are not quotable")
endif()

target_include_directories(browser_llm_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

bllm_embed_shaders(
    TARGET  browser_llm_core
    OUTPUT  "${CMAKE_BINARY_DIR}/generated/bllm/shaders_generated.h"
    SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/vector_add.wgsl"
)

if(EMSCRIPTEN)
    # These need an implementation of webgpu.h. In the wasm configuration that
    # comes from --use-port=emdawnwebgpu. Natively it would come from Dawn,
    # which is not linked yet: Dawn arrives with the first model kernel, where
    # correctness needs a CPU reference to check against. Until then these
    # sources build for wasm only, and the native test binary covers the
    # platform-neutral logic.
    target_sources(browser_llm_core PRIVATE
        src/core/gpu/device.cpp
        src/core/gpu/self_check.cpp
    )

    # Not merely preprocessed away — not compiled at all in a clean build, so
    # the symbol cannot reach the shipped module.
    if(BLLM_ENABLE_DIAGNOSTICS)
        target_sources(browser_llm_core PRIVATE src/core/gpu/readback_bench.cpp)
    endif()

    # -----------------------------------------------------------------------
    # Browser target. The only Emscripten-aware translation unit.
    # -----------------------------------------------------------------------
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/wasm/bindings.cpp")
        message(FATAL_ERROR
            "src/wasm/bindings.cpp does not exist yet.\n"
            "BLLM-001 is still implementing: the WebGPU device path has not "
            "been written, because it has not yet been compiled against a real "
            "Emscripten toolchain. See "
            "docs/decisions/packets/2026-08-29-repo-skeleton-and-build-system.md")
    endif()

    add_executable(browser_llm src/wasm/bindings.cpp)
    target_link_libraries(browser_llm PRIVATE browser_llm_core)

    target_compile_options(browser_llm_core PUBLIC --use-port=emdawnwebgpu)
    target_link_options(browser_llm PRIVATE
        --use-port=emdawnwebgpu
        -sMODULARIZE=1
        -sEXPORT_ES6=1
        -sENVIRONMENT=web,worker
        -sALLOW_MEMORY_GROWTH=1
        -sEXPORTED_RUNTIME_METHODS=['ccall','cwrap']
        # Deliberately absent: -pthread. GitHub Pages cannot set COOP/COEP,
        # so SharedArrayBuffer is unavailable and pthreads cannot be used.
    )
    set_target_properties(browser_llm PROPERTIES
        OUTPUT_NAME "browser_llm"
        SUFFIX ".mjs"
    )
else()
    # -----------------------------------------------------------------------
    # Native: tests only. No browser, no GPU required at this stage.
    # -----------------------------------------------------------------------
    enable_testing()
    add_subdirectory(tests)
endif()

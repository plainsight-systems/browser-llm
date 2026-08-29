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
add_library(browser_llm_core STATIC
    src/core/gpu/dispatch_math.cpp
)

target_include_directories(browser_llm_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

bllm_embed_shaders(
    TARGET  browser_llm_core
    OUTPUT  "${CMAKE_BINARY_DIR}/generated/bllm/shaders_generated.h"
    SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/vector_add.wgsl"
)

if(EMSCRIPTEN)
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

cmake_minimum_required(VERSION 3.25)
project(charlotte LANGUAGES CXX)

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

add_library(charlotte_core STATIC
    src/core/arch/describe.cpp
    src/core/arch/gemma3/gemma3.cpp
    src/core/arch/llama/llama.cpp
    src/core/arch/qwen3/qwen3.cpp
    src/core/capability/capability.cpp
    src/core/gguf/arrays.cpp
    src/core/gguf/index.cpp
    src/core/gguf/reader.cpp
    src/core/preflight/preflight.cpp
    src/core/tokenizer/bpe/byte_level_bpe.cpp
    src/core/tokenizer/bpe/merge.cpp
    src/core/tokenizer/bpe/merge_table.cpp
    src/core/tokenizer/bpe/piece_cache.cpp
    src/core/tokenizer/load.cpp
    src/core/tokenizer/nfc.cpp
    src/core/tokenizer/pretokenize.cpp
    src/core/tokenizer/special.cpp
    src/core/tokenizer/unicode.cpp
    src/core/tokenizer/unicode_tables.cpp
    src/core/tokenizer/utf8_stream.cpp
    src/core/tokenizer/vocabulary.cpp
    src/core/residency/plan.cpp
    src/core/gpu/dispatch_math.cpp
)

if(BLLM_ENABLE_DIAGNOSTICS)
    target_compile_definitions(charlotte_core PUBLIC BLLM_DIAGNOSTICS_ENABLED=1)
    message(STATUS
        "charlotte: diagnostics ENABLED — timings from this build are not quotable")
endif()

target_include_directories(charlotte_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

bllm_embed_shaders(
    TARGET  charlotte_core
    OUTPUT  "${CMAKE_BINARY_DIR}/generated/bllm/shaders_generated.h"
    SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/vector_add.wgsl"
)

if(EMSCRIPTEN)
    # These need an implementation of webgpu.h. In the wasm configuration that
    # comes from --use-port=emdawnwebgpu. The native configuration does not
    # link Dawn, the other implementation, because nothing it builds needs a
    # GPU: these sources build for wasm only, and the native test binary
    # covers the platform-neutral logic.
    target_sources(charlotte_core PRIVATE
        src/core/gpu/device.cpp
        src/core/gpu/self_check.cpp
    )

    # Not merely preprocessed away — not compiled at all in a clean build, so
    # the symbol cannot reach the shipped module.
    if(BLLM_ENABLE_DIAGNOSTICS)
        target_sources(charlotte_core PRIVATE src/core/gpu/readback_bench.cpp)
    endif()

    # -----------------------------------------------------------------------
    # Browser target. The only Emscripten-aware translation unit.
    # -----------------------------------------------------------------------
    add_executable(charlotte src/wasm/bindings.cpp)
    target_link_libraries(charlotte PRIVATE charlotte_core)

    target_compile_options(charlotte_core PUBLIC --use-port=emdawnwebgpu)
    target_link_options(charlotte PRIVATE
        --use-port=emdawnwebgpu
        -sMODULARIZE=1
        -sEXPORT_ES6=1
        -sENVIRONMENT=web,worker
        # WASM.1: growth is a FALLBACK, not a default. It converts a
        # deterministic out-of-memory at startup into a nondeterministic one
        # under load. It is enabled because -sINITIAL_HEAP is sized only from
        # the loader's measured peak, never from a figure chosen for comfort,
        # and no measured peak exists to size it from.
        #
        # MAXIMUM_MEMORY bounds the damage and makes an out-of-memory
        # reproducible. The default is 2GB, which is a ceiling inherited rather
        # than chosen.
        -sALLOW_MEMORY_GROWTH=1
        -sMAXIMUM_MEMORY=536870912
        -sEXPORTED_RUNTIME_METHODS=['HEAPU8']
        -sEXPORTED_FUNCTIONS=['_main','_malloc','_free']
        # Deliberately absent: -pthread. GitHub Pages cannot set COOP/COEP,
        # so SharedArrayBuffer is unavailable and pthreads cannot be used.
    )
    set_target_properties(charlotte PROPERTIES
        OUTPUT_NAME "charlotte"
        SUFFIX ".mjs"
    )
else()
    # -----------------------------------------------------------------------
    # Native: tests only. No browser, no GPU required at this stage.
    # -----------------------------------------------------------------------
    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(bench)
endif()

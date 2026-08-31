include(FetchContent)

# Pinned by tag. Dependency resolution must be deterministic; a floating ref
# would make the test build a moving target.
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        1da23a3e8119ec5cce4f9388e91b065e20bf06f5
)
FetchContent_MakeAvailable(doctest)

add_executable(browser_llm_tests
    dispatch_math_test.cpp
    shader_embed_test.cpp
    unique_handle_test.cpp
    run_guard_test.cpp
    main_test.cpp
)

target_link_libraries(browser_llm_tests PRIVATE browser_llm_core doctest::doctest)

# The shader test reads the on-disk .wgsl to compare against the embedded copy.
target_compile_definitions(browser_llm_tests PRIVATE
    BLLM_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../src/shaders"
)

add_test(NAME browser_llm_tests COMMAND browser_llm_tests)

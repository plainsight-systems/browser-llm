# GPU tests: core code that holds WebGPU objects, run natively on Dawn. Locally
# that is this machine's GPU; in CI, Mesa's software Vulkan driver. With no
# adapter at all the tests fail, never skip: a GPU test that passes because it
# ran nothing would claim what was never checked.
add_executable(charlotte_gpu_tests
    device_test.cpp
    mapping_test.cpp
    upload_test.cpp
    unpack_f32_test.cpp
    unpack_q4_0_test.cpp
    unpack_q4_1_test.cpp
    unpack_q6_k_test.cpp
    ../main_test.cpp
)
if(BLLM_ENABLE_DIAGNOSTICS)
    target_sources(charlotte_gpu_tests PRIVATE upload_check_test.cpp)
endif()
target_link_libraries(charlotte_gpu_tests PRIVATE charlotte_gpu doctest::doctest)
target_include_directories(charlotte_gpu_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_compile_definitions(charlotte_gpu_tests PRIVATE
    BLLM_GGUF_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../fixtures/gguf")

add_test(NAME charlotte_gpu_tests COMMAND charlotte_gpu_tests)

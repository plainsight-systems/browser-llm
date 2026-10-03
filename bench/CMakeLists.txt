# Tokenizer throughput on a pinned corpus (make bench). Not built by default:
# a benchmark is run on purpose, in a release build, never as part of a test.
add_executable(charlotte_bench_tokenizer EXCLUDE_FROM_ALL tokenizer_bench.cpp)
target_link_libraries(charlotte_bench_tokenizer PRIVATE charlotte_core)
# headers.inc names each model's fetched header and its file's size.
target_include_directories(charlotte_bench_tokenizer PRIVATE "${CMAKE_SOURCE_DIR}/tests")
target_compile_definitions(charlotte_bench_tokenizer PRIVATE
    BLLM_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/.cache/test-data"
)

# Upload's CPU stage on each quantized layout (make bench).
add_executable(charlotte_bench_piece_writer EXCLUDE_FROM_ALL piece_writer_bench.cpp)
target_link_libraries(charlotte_bench_piece_writer PRIVATE charlotte_core)

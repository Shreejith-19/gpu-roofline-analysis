add_executable(smoke_test smoke_test.cu)

target_include_directories(smoke_test PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(smoke_test PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

target_link_libraries(smoke_test PRIVATE gpu_roofline_kernels)
add_test(NAME CudaSmokeTest COMMAND smoke_test)

# Baseline Naive Matmul Unit & Verification Tests
add_executable(test_matmul_naive test_matmul_naive.cu)

target_include_directories(test_matmul_naive PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_matmul_naive PRIVATE
    gpu_roofline_kernels
)

set_target_properties(test_matmul_naive PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

add_test(NAME TestMatmulNaive COMMAND test_matmul_naive)

# Shared-Memory Tiled Matmul Unit & Verification Tests
add_executable(test_matmul_tiled test_matmul_tiled.cu)

target_include_directories(test_matmul_tiled PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_matmul_tiled PRIVATE
    gpu_roofline_kernels
)

set_target_properties(test_matmul_tiled PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

add_test(NAME TestMatmulTiled COMMAND test_matmul_tiled)

# Parallel Sum Reduction Unit & Verification Tests
add_executable(test_reduce_sum test_reduce_sum.cu)

target_include_directories(test_reduce_sum PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_reduce_sum PRIVATE
    gpu_roofline_kernels
)

set_target_properties(test_reduce_sum PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

add_test(NAME TestReduceSum COMMAND test_reduce_sum)

# Parallel Max Reduction Unit & Verification Tests
add_executable(test_reduce_max test_reduce_max.cu)

target_include_directories(test_reduce_max PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_reduce_max PRIVATE
    gpu_roofline_kernels
)

set_target_properties(test_reduce_max PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

add_test(NAME TestReduceMax COMMAND test_reduce_max)



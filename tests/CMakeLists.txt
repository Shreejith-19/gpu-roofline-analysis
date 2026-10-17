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

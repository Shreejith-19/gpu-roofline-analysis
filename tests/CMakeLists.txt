add_executable(smoke_test smoke_test.cu)

target_include_directories(smoke_test PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(smoke_test PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

add_test(NAME CudaSmokeTest COMMAND smoke_test)

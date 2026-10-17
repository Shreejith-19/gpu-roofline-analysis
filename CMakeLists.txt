cmake_minimum_required(VERSION 3.24)

project(gpu_roofline_analysis
    VERSION 0.1.0
    DESCRIPTION "CUDA GPU Performance Modeling and Empirical Roofline Analysis"
    LANGUAGES CXX CUDA
)

# ---------------------------------------------------------
# C++ and CUDA Language Standards
# ---------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ---------------------------------------------------------
# CUDA Architecture Configuration
# ---------------------------------------------------------
# Defaults to Ampere SM 86 (RTX 3050 Laptop / Desktop).
# Override from command line: -DCMAKE_CUDA_ARCHITECTURES="86" or "native"
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "Target CUDA architecture(s)" FORCE)
endif()
message(STATUS "[Roofline] Target CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")

# ---------------------------------------------------------
# Compiler Flags
# ---------------------------------------------------------
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    )
    # CUDA flags passed through host compiler
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
endif()

# ---------------------------------------------------------
# Common Includes
# ---------------------------------------------------------
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

# ---------------------------------------------------------
# Testing
# ---------------------------------------------------------
option(BUILD_TESTS "Build test suite" ON)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

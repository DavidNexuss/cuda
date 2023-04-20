cmake_minimum_required(VERSION 3.9.4)
# Default CMAKE Configuration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Project setup
project(TGA LANGUAGES CUDA C)
enable_language(CUDA)

file(GLOB_RECURSE SRC src/*.c)
add_executable(tga ${SRC})

# Library setup
add_subdirectory(lib/stb)
target_link_libraries(tga PRIVATE stb)

# Cuda setup
include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set_target_properties(tga PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# LTO
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

if( supported )
    message(STATUS "IPO / LTO enabled")
    set_property(TARGET tga PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(STATUS "IPO / LTO not supported: <${error}>")
endif()

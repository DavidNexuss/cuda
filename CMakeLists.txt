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

find_package(CUDA)
set(CMAKE_CUDA_STANDARD 14)

file(GLOB_RECURSE SRC src/*.c src/*.cu)
add_library(tga ${SRC})

# Library setup
target_link_libraries(tga PRIVATE cuda)
target_include_directories(tga PUBLIC /opt/cuda/include/ src include)

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


add_executable(test1 scripts/test1.c scripts/runFrame.c)
target_include_directories(test1 PRIVATE include)
target_link_libraries(test1 tga cuda)

add_executable(test2 scripts/test2.c scripts/runFrame.c)
target_include_directories(test2 PRIVATE include)
target_link_libraries(test2 tga cuda)

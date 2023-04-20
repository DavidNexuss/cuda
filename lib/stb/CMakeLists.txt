cmake_minimum_required(VERSION 3.6)
project(stb)
file(GLOB SRC *.c)
add_library(stb ${SRC})

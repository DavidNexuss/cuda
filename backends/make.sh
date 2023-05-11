#!/bin/sh
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd $SCRIPT_DIR
cd thirdparty/assimp
cmake . -B build
cd build
make -j4
cd ../../..
nvcc -g -c loader.cpp -I ../include
ar rcs backend.a loader.o

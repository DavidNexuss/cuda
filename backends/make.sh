#!/bin/sh
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd $SCRIPT_DIR
cd thirdparty/assimp
cmake . -B build
cd build
make -j4
cd ../../..
make

#!/bin/sh
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
rm -rf *.o *.a
rm -rf thirdparty/assimp/build

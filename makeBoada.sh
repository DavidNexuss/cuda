#!/bin/sh
mkdir -p obj bin
export CUDA_HOME=/Soft/cuda/11.2.1
make all -j4 -f MakefileBoada

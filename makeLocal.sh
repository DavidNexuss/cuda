#!/bin/sh
mkdir -p obj bin
export CUDA_HOME=/opt/cuda/
make all

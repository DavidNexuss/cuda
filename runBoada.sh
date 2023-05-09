#!/bin/sh
./makeBoada.sh
export CUDA_HOME=/Soft/cuda/11.2.1
INCLUDE_DIRS="-I$CUDA_HOME/include -I$CUDA_HOME/sdk/CUDALibraries/common/inc -I include -I src -I$CUDA_HOME/sdk/CUDALibraries/common/lib -I/opt/cuda/include -Iinclude"
nvcc -O3 -Wno-deprecated-gpu-targets -gencode arch=compute_86,code=sm_86 --ptxas-options=-v   -DIMPL $@ tracer.a -lcudart -Xlinker -rpath,$CUDA_HOME/lib64 $INCLUDE_DIRS -o traceProgram -Xcompiler -fPIC

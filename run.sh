#!/bin/sh
./makeLocal.sh
CUDA_HOME="/opt/cuda"
INCLUDE_DIRS="-I$CUDA_HOME/include -I$CUDA_HOME/sdk/CUDALibraries/common/inc -I include -I src -I$CUDA_HOME/sdk/CUDALibraries/common/lib -I/opt/cuda/include -Iinclude"
nvcc -DIMPL $@ tracer.a -lcudart -Xlinker -rpath,$CUDA_HOME/lib64 $INCLUDE_DIRS -o traceProgram -Xcompiler -fopenmp
./traceProgram
rm -f traceProgram

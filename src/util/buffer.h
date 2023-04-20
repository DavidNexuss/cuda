#pragma once
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

// MUTABLE BUFFER, abstraction to handle pairs of host and device buffers
typedef struct {
  unsigned char *H, *D;
  unsigned int   allocatedSize;
} Buffer;

inline Buffer bufferCreate(int size) {
  Buffer buffer;
  buffer.H = (unsigned char*)malloc(size);
  cudaMalloc((void**)&buffer.D, size);
  return buffer;
}

inline void bufferDestroy(Buffer* buffer) {
  free(buffer->H);
  cudaFree((void*)buffer->D);
}

inline void bufferUpload(Buffer* buffer) {
  cudaMemcpy((void*)buffer->D, (void*)buffer->H, buffer->allocatedSize, cudaMemcpyHostToDevice);
}

inline void bufferDownload(Buffer* buffer) {
  cudaMemcpy((void*)buffer->H, (void*)buffer->D, buffer->allocatedSize, cudaMemcpyDeviceToHost);
}

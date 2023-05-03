#pragma once
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

// MUTABLE BUFFER, abstraction to handle pairs of host and device buffers

// For error checking and correct memory release
static int gBufferTotalAllocatedSize = 0;
static int gBufferPeakAllocatedSize  = 0;

typedef struct {
  unsigned char *H, *D;
  unsigned int   allocatedSize;
} Buffer;

inline Buffer bufferCreate(int size) {
  gBufferTotalAllocatedSize += size;
  gBufferPeakAllocatedSize = gBufferTotalAllocatedSize > gBufferPeakAllocatedSize ? gBufferTotalAllocatedSize : gBufferPeakAllocatedSize;

  Buffer buffer;
  buffer.allocatedSize = size;
  buffer.H             = (unsigned char*)malloc(size);
  cudaMalloc((void**)&buffer.D, size);
  return buffer;
}

inline void bufferDestroy(Buffer* buffer) {
  gBufferTotalAllocatedSize -= buffer->allocatedSize;
  free(buffer->H);
  cudaFree((void*)buffer->D);
}

inline void bufferUpload(Buffer* buffer, int amount = -1) {
  if (amount < 0) {
    amount = buffer->allocatedSize;
  }
  if (amount > buffer->allocatedSize) {
    //TODO: Place warning
  }
  cudaMemcpy((void*)buffer->D, (void*)buffer->H, buffer->allocatedSize, cudaMemcpyHostToDevice);
}

inline void bufferDownload(Buffer* buffer) {
  cudaMemcpy((void*)buffer->H, (void*)buffer->D, buffer->allocatedSize, cudaMemcpyDeviceToHost);
}

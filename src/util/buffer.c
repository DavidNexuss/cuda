#include "buffer.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

// MUTABLE BUFFER, abstraction to handle pairs of host and device buffers

// For error checking and correct memory release
int gBufferTotalAllocatedSize = 0;
int gBufferPeakAllocatedSize  = 0;


Buffer bufferCreate(int size) {
  gBufferTotalAllocatedSize += size;
  gBufferPeakAllocatedSize = gBufferTotalAllocatedSize > gBufferPeakAllocatedSize ? gBufferTotalAllocatedSize : gBufferPeakAllocatedSize;

  Buffer buffer;
  buffer.allocatedSize = size;
  buffer.H             = (unsigned char*)malloc(size);
  cudaMalloc((void**)&buffer.D, size);
  return buffer;
}
void* bufferCreateImmutable(void* data, int size) {
  void* buffer;
  cudaMalloc((void**)&buffer, size);
  cudaMemcpy((void*)buffer, data, size, cudaMemcpyHostToDevice);
  return buffer;
}

void bufferDestroy(Buffer* buffer) {
  gBufferTotalAllocatedSize -= buffer->allocatedSize;
  if (buffer->H != 0) free(buffer->H);
  if (buffer->D != 0) cudaFree((void*)buffer->D);
}

void bufferDestroyImmutable(void* buffer) { cudaFree(buffer); }

#include <stdio.h>
void bufferUploadAmount(Buffer* buffer, int amount) {
  if(amount < -1) { 
    dprintf(2, "Probably doing somehting wrong");
    exit(1);
  }
  if (amount < 0) {
    amount = buffer->allocatedSize;
  }
  if (amount > buffer->allocatedSize) {
    dprintf(2, "Too much memory to upload");
    exit(1);
  }
  cudaMemcpy((void*)buffer->D, (void*)buffer->H, buffer->allocatedSize, cudaMemcpyHostToDevice);
}

void bufferUpload(Buffer* buffer) { bufferUploadAmount(buffer, -1); }

void bufferDownload(Buffer* buffer) {
  cudaMemcpy((void*)buffer->H, (void*)buffer->D, buffer->allocatedSize, cudaMemcpyDeviceToHost);
}

void bufferDebugStats() {
  dprintf(2, "[STATS] Peak memory use: %d\n", gBufferPeakAllocatedSize);
  dprintf(2, "[STATS] Memory leak : %d\n", gBufferTotalAllocatedSize);
}

#include "buffer.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

// MUTABLE BUFFER, abstraction to handle pairs of host and device buffers

#define KILL dprintf(2, "%d", 0 / 0)
// For error checking and correct memory release
int gBufferTotalAllocatedSize = 0;
int gBufferPeakAllocatedSize  = 0;


Buffer bufferCreate(int size) {
  gBufferTotalAllocatedSize += size;
  gBufferPeakAllocatedSize = gBufferTotalAllocatedSize > gBufferPeakAllocatedSize ? gBufferTotalAllocatedSize : gBufferPeakAllocatedSize;

  Buffer buffer;
  buffer.allocatedSize = size;
  buffer.H             = (unsigned char*)malloc(size);
  if(cudaMalloc((void**)&buffer.D, size) != cudaSuccess) { 
	dprintf(2, "Error cuda malloc regular buffer %d\n", size);
	KILL;
  }
  return buffer;
}
void* bufferCreateImmutable(void* data, unsigned long long size) {
  void* buffer;
  if(cudaMalloc((void**)&buffer, size) != cudaSuccess) { 
	dprintf(2, "Error cuda malloc immutable buffer %ldd\n", size);
	KILL;
  }
  cudaMemcpy((void*)buffer, data, size, cudaMemcpyHostToDevice);
  dprintf(2, "Create immutable %p\n", buffer);
  return buffer;
}

void bufferDestroy(Buffer* buffer) {
  gBufferTotalAllocatedSize -= buffer->allocatedSize;
  if (buffer->H != 0) free(buffer->H);
  if (buffer->D != 0) {
    if (cudaFree((void*)buffer->D) != cudaSuccess) {
      dprintf(2, "Cuda free error %p!\n", buffer);
      exit(1);
    }
  }
}

void bufferDestroyImmutable(void* buffer) {
  if (buffer == 0) return;
  if (cudaFree(buffer) != cudaSuccess) {
    dprintf(2, "Cuda free immutable error %p!\n", buffer);
    KILL;
  }
}

void bufferUploadAmount(Buffer* buffer, int amount) {
  if (amount < -1) {
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

  dprintf(2, " ----> uploading %p %d\n", buffer, amount);
  cudaMemcpy((void*)buffer->D, (void*)buffer->H, buffer->allocatedSize, cudaMemcpyHostToDevice);
}

void bufferUpload(Buffer* buffer) {
  bufferUploadAmount(buffer, -1);
}

void bufferDownload(Buffer* buffer) {
  dprintf(2, " <----- downloading %p %lld\n", buffer, buffer->allocatedSize);
  cudaMemcpy((void*)buffer->H, (void*)buffer->D, buffer->allocatedSize, cudaMemcpyDeviceToHost);
}

void bufferDebugStats() {
  dprintf(2, "[STATS] Peak memory use: %d\n", gBufferPeakAllocatedSize);
  dprintf(2, "[STATS] Memory leak : %d\n", gBufferTotalAllocatedSize);
}

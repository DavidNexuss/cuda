#pragma once

/* Simple struct to keep synced dynamic allocated buffers between CPU and GPU */
typedef struct
{
  void *             H, *D;
  unsigned long long allocatedSize;
} Buffer;


Buffer bufferCreate(int size);
void*  bufferCreateImmutable(void* data, unsigned long long size);
void   bufferDestroy(Buffer* buffer);
void   bufferDestroyImmutable(void* buffer);
void   bufferUploadAmount(Buffer* buffer, int amount);
void   bufferUpload(Buffer* buffer);
void   bufferDownload(Buffer* buffer);

void bufferDebugStats();

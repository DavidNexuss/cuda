#pragma once

typedef struct {
  void *H, *D;
  unsigned int   allocatedSize;
} Buffer;

Buffer bufferCreate(int size);
void*  bufferCreateImmutable(void* data, int size);
void   bufferDestroy(Buffer* buffer);
void   bufferDestroyImmutable(void* buffer);
void   bufferUploadAmount(Buffer* buffer, int amount);
void   bufferUpload(Buffer* buffer);
void   bufferDownload(Buffer* buffer);

void bufferDebugStats();

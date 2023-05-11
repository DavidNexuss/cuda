#pragma once
#ifdef __cplusplus
extern "C" {
#endif
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
#ifdef __cplusplus
}
#endif

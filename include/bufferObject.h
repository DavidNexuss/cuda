#pragma once
#ifdef __cplusplus
extern "C" {
#endif
typedef struct
{
  void* data;
  int   count;
} BufferObject;

typedef struct
{
  BufferObject vertexBuffer;
  BufferObject indexBuffer;
} ObjResult;

ObjResult createBufferObject(const char*);
#ifdef __cplusplus
}
#endif

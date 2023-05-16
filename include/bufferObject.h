#pragma once


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

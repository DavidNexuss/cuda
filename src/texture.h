#pragma once
#include "util/buffer.h"

typedef struct {
  int   width;
  int   height;
  int   channels;
  void* data;
} Texture;


Texture textureCreate(const char* texturePath);
void    textureDestroy(Texture* text);

typedef struct { 
  void* data;
  int count;
} BufferObject;

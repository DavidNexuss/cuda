#pragma once
#include "util/buffer.h"

typedef struct {
  int    width;
  int    height;
  int    channels;
  Buffer data;
} Texture;


Texture textureCreate(const char* texturePath);
void    textureDestroy(Texture* text);
void    textureUpload(Texture* text);

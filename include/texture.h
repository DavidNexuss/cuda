#pragma once
#include "util/buffer.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct
{
  int   width;
  int   height;
  int   channels;
  void* data;
} Texture;


Texture textureCreate(const char* texturePath);
void    textureDestroy(Texture* text);
#ifdef __cplusplus
}
#endif

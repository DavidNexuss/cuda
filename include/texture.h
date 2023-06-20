#pragma once
#include "util/buffer.h"


/* Texture definiton struct self explanatory */
typedef struct
{
  int   width;
  int   height;
  int   channels;
  void* data;
} Texture;


Texture textureCreate(const char* texturePath);
void    textureDestroy(Texture* text);

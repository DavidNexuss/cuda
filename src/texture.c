#include "texture.h"
#include "util/buffer.h"
#include <stb/stb_image.h>
#include <string.h>
Texture textureCreate(const char* texturePath) {
  Texture        text;
  unsigned char* data = stbi_load(texturePath, &text.width, &text.height, &text.channels, 3);
  if(data == 0) { 
    dprintf(2, "[IO] Error trying to load texture %s\n", texturePath);
    exit(1);
  } 

  dprintf(2, "[IO] Texture %s load successfully\n", texturePath);
  text.data           = bufferCreateImmutable(data, text.width * text.height * text.channels * sizeof(char));
  stbi_image_free(data);
  return text;
}
void textureDestroy(Texture* text) {
  bufferDestroyImmutable(text->data);
}

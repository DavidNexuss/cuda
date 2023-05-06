#include "texture.h"
#include "util/buffer.h"
#include <stb/stb_image.h>
#include <string.h>

Texture textureCreate(const char* texturePath) {
  Texture        text;
  text.data = stbi_load(texturePath, &text.width, &text.height, &text.channels, 3);
  if(text.data <= 0) { 
    dprintf(2, "[IO] Error trying to load texture %s\n", texturePath);
    exit(1);
  } 
  return text;
}
void textureDestroy(Texture* text) {
  stbi_image_free(text->data);
}

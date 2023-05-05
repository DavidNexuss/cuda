#include "texture.h"
#include "util/buffer.h"
#include <stb/stb_image.h>
#include <string.h>
Texture textureCreate(const char* texturePath) {
  Texture        text;
  unsigned char* data = stbi_load(texturePath, &text.width, &text.height, &text.channels, 3);
  text.data           = bufferCreateImmutable(data, text.width * text.height * text.channels * sizeof(char));
  stbi_image_free(data);
  return text;
}
void textureDestroy(Texture* text) {
  bufferDestroyImmutable(&text->data);
}

#include "texture.h"
#include "util/buffer.h"
#include <stb/stb_image.h>
#include <string.h>
Texture textureCreate(const char* texturePath) {
  Texture        text;
  unsigned char* data = stbi_load(texturePath, &text.width, &text.height, &text.channels, 3);
  text.data           = bufferCreate(text.width * text.height * text.channels);
  memcpy(text.data.H, data, text.data.allocatedSize);
  stbi_image_free(data);
  return text;
}
void textureDestroy(Texture* text) {
  bufferDestroy(&text->data);
}
void textureUpload(Texture* text) {
  bufferUpload(&text->data);
}

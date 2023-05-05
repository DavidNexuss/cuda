#include "scene.h"
#include "texture.h"
#include "util/buffer.h"
#include <stb/stb_image_write.h>
Scene sceneCreate(SceneDesc desc) {
  Scene scene;
  scene.desc        = desc;
  scene.textures    = bufferCreate(sizeof(Texture) * desc.maxTextures);
  scene.meshes      = bufferCreate(sizeof(Mesh) * desc.maxMeshes);
  scene.materials   = bufferCreate(sizeof(Material) * desc.maxMaterials);
  scene.framebuffer = bufferCreate(3 * sizeof(float) * desc.frameBufferWidth * desc.frameBufferHeight * desc.framesInFlight);
  scene.constants   = bufferCreate(sizeof(PushConstants) * desc.framesInFlight);

  scene.materialCount = -1;
  scene.textureCount  = -1;
  scene.meshCount     = -1;
  scene.indexBufferCount = -1;
  scene.vertexBufferCount = -1;

  scene.objects = (Buffer*)malloc(sizeof(Buffer) * desc.framesInFlight);

  PushConstants* cn = sceneInputHost(&scene).constants;
  for (int i = 0; i < desc.framesInFlight; i++) {
    scene.objects[i] = bufferCreate(sizeof(Object) * desc.maxObjects);
    cn[i].objects    = (Object*)scene.objects[i].H;
  }

  // Late
  scene.vertexBuffers = (void**)malloc(sizeof(void*) * desc.maxVertexBuffer);
  scene.indexBuffers  = (void**)malloc(sizeof(void*) * desc.maxIndexBuffer);

  return scene;
}

/* returns host bag of pointers */
SceneInput sceneInputHost(Scene* scene) {
  SceneInput inp;
  inp.materials = (Material*)scene->materials.H;
  inp.meshes    = (Mesh*)scene->meshes.H;
  inp.textures  = (Texture*)scene->textures.H;
  inp.constants = (PushConstants*)scene->constants.H;
  return inp;
}

SceneInput sceneInputDevice(Scene* scene) {
  SceneInput inp;
  inp.materials = (Material*)scene->materials.D;
  inp.meshes    = (Mesh*)scene->meshes.D;
  inp.textures  = (Texture*)scene->textures.D;
  inp.constants = (PushConstants*)scene->constants.D;
  return inp;
}

/* destroys scene and releases memory */
void sceneDestroy(Scene* scene) {
  bufferDestroy(&scene->meshes);
  bufferDestroy(&scene->materials);
  bufferDestroy(&scene->framebuffer);

  for (int i = 0; i < scene->textureCount; i++) {
    textureDestroy(&((Texture*)scene->textures.H)[i]);
  }
  bufferDestroy(&scene->textures);

  for (int i = 0; i < scene->desc.framesInFlight; i++) {
    bufferDestroy(&scene->objects[i]);
  }

  bufferDestroy(&scene->constants);
  free(scene->objects);

  //Late

  for (int i = 0; i < scene->vertexBufferCount; i++) {
    bufferDestroyImmutable(&scene->vertexBuffers[i]);
  }

  for (int i = 0; i < scene->indexBufferCount; i++) {
    bufferDestroyImmutable(&scene->indexBuffers[i]);
  }

  free(scene->vertexBuffers);
  free(scene->indexBuffers);
}

/* uploads scene */
void sceneUpload(Scene* scene) {
  bufferUploadAmount(&scene->materials, scene->materialCount * sizeof(Material));
  bufferUploadAmount(&scene->meshes, scene->meshCount * sizeof(Mesh));
  bufferUploadAmount(&scene->textures, scene->textureCount * sizeof(Texture));
}

void sceneUploadObjects(Scene* scene) {
  //Upload required buffer objects for each frame
  PushConstants* c = sceneInputHost(scene).constants;
  {
    for (int i = 0; i < scene->desc.framesInFlight; i++) {
      bufferUploadAmount(&scene->objects[i], c[i].objectCount * sizeof(Object));
    }
  }

  //Upload push constants for each frame
  {
    for (int i = 0; i < scene->desc.framesInFlight; i++) {
      c[i].objects = (Object*)scene->objects[i].D;
    }
    bufferUpload(&scene->constants);

    for (int i = 0; i < scene->desc.framesInFlight; i++) {
      c[i].objects = (Object*)scene->objects[i].H;
    }
  }
}

PushConstants* scenePushConstantsHost(Scene* scene) {
  return (PushConstants*)scene->constants.H;
}
PushConstants* scenePushConstantsDevice(Scene* scene) {
  return (PushConstants*)scene->constants.D;
}

void sceneDownload(Scene* scene) {
  bufferDownload(&scene->framebuffer);
}

float* sceneGetFrame(Scene* scene, int index) {
  float* fbo = (float*)scene->framebuffer.H;
  return &fbo[index * 3 * scene->desc.frameBufferWidth * scene->desc.frameBufferHeight];
}

#include <stdio.h>
void sceneWriteFrame(Scene* scene, const char* path, int index) {
  dprintf(2, "[IO] Writing frame [%d] for scene to %s\n", index, path);
  if(scene->desc.fWriteClamped) { 
    
    int count = scene->desc.frameBufferWidth * scene->desc.frameBufferHeight * 3;
    float* fbo = sceneGetFrame(scene, index);
    unsigned char* png = (unsigned char*)malloc(count);

    float maxValue = 0.0f;
    float minValue = 10000000.0f;
    for(int i = 0; i < count; i++) { 
      if(fbo[i] > maxValue) maxValue = fbo[i];
      if(fbo[i] < minValue) minValue = fbo[i];
    }
    
    dprintf(2, "Max and min values %f %f\n", maxValue, minValue);
    if((maxValue - minValue) < 0.01) { maxValue += 0.2; }
    for(int i = 0; i < count; i++) { 
      png[i] = ((fbo[i] - minValue)/ (maxValue - minValue)) * 0xff;
    }
    
    stbi_write_png(path, scene->desc.frameBufferWidth, scene->desc.frameBufferHeight, 3, png, scene->desc.frameBufferWidth * 3);
    free(png);
  } else { 
    stbi_write_hdr(path, scene->desc.frameBufferWidth, scene->desc.frameBufferHeight, 3, sceneGetFrame(scene, index)); 
  }
}

#include <stdio.h>
#include "util/buffer.h"
#include "trace.h"

//SCENE
typedef struct {
  int maxObjects;
  int maxMaterials;
  int maxMeshes;
  int frameBufferWidth;
  int frameBufferHeight;

} SceneDesc;

typedef struct {
  Buffer meshes;
  Buffer objects;
  Buffer materials;
  Buffer framebuffer;

  SceneDesc desc;
  int       objectCount;

} Scene;

typedef struct {
  Object*   objects;
  Material* materials;
  Mesh*     meshes;
} SceneInit;

Scene sceneCreate(SceneDesc desc) {
  Scene scene;
  scene.desc        = desc;
  scene.meshes      = bufferCreate(sizeof(Mesh) * desc.maxMeshes);
  scene.objects     = bufferCreate(sizeof(Object) * desc.maxObjects);
  scene.materials   = bufferCreate(sizeof(Material) * desc.maxMaterials);
  scene.framebuffer = bufferCreate(3 * sizeof(float) * desc.frameBufferWidth * desc.frameBufferHeight);
  return scene;
}

SceneInit sceneInit(Scene* scene, int objectCount) {
  scene->objectCount = objectCount;
  return {
    (Object*)scene->objects.H,
    (Material*)scene->materials.H,
    (Mesh*)scene->meshes.H};
}

void sceneDestroy(Scene* scene) {
  bufferDestroy(&scene->meshes);
  bufferDestroy(&scene->objects);
  bufferDestroy(&scene->materials);
  bufferDestroy(&scene->framebuffer);
}

void sceneUpload(Scene* scene) {
  bufferUpload(&scene->materials);
  bufferUpload(&scene->objects);
  bufferUpload(&scene->meshes);
}

void sceneDownload(Scene* scene) {
  bufferDownload(&scene->framebuffer);
}

int main(int argc, char** argv) {
  SceneDesc sceneDesc = {};
  Scene     scene     = sceneCreate(sceneDesc);

  {
    auto init = sceneInit(&scene, 2);
    sceneUpload(&scene);
  }



  sceneDestroy(&scene);
}

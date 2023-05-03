#include <stdio.h>
#include <stb/stb_image_write.h>
#include "texture.h"
#include "util/buffer.h"
#include "objects.h"
#include "scene.h"



void defaultScene(Scene* scene);
int  defaultSceneLoop(Object* objects, float t);

void programRun(const char* path, int width, int height, void(initScene)(Scene*), int(initSceneFrame)(Object*, float t)) {

  SceneDesc sceneDesc           = {};
  sceneDesc.maxMeshes           = 300;
  sceneDesc.maxObjects          = 400;
  sceneDesc.maxMaterials        = 300;
  sceneDesc.frameBufferWidth    = width;
  sceneDesc.frameBufferHeight   = height;
  sceneDesc.numThreads          = 4;
  sceneDesc.iterationsPerThread = 4;
  sceneDesc.rayDepth            = 4;
  sceneDesc.framesInFlight      = 1;
  sceneDesc.frameDelta          = 0.1;

  Scene scene = sceneCreate(sceneDesc);

  //Inits scene materials and meshes
  {
    initScene(&scene);
    sceneUpload(&scene);
  }

  //Inits scene objects
  {
    float   t   = 0;
    Object* src = (Object*)scene.objects.H;

    for (int i = 0; i < sceneDesc.framesInFlight; i++) {
      int objects = initSceneFrame((Object*)scene.objects.H, t);
      t += sceneDesc.frameDelta;
      scene.objectCount = objects;
      src += objects;
    }

    sceneUploadObjects(&scene);
  }

  sceneRun(&scene);
  sceneDownload(&scene);
  sceneWriteFrame(&scene, path, 0);
  sceneDestroy(&scene);
}

int main(int argc, char** argv) {

  programRun("result.hdr", 1024, 1024, defaultScene, defaultSceneLoop);
  programRun("result2.hdr", 1024 * 2, 1024 * 2, defaultScene, defaultSceneLoop);
  programRun("result.hdr", 1024, 1024, defaultScene, defaultSceneLoop);
  programRun("result2.hdr", 1024 * 2, 1024 * 2, defaultScene, defaultSceneLoop);
  programRun("result3.hdr", 1024 * 4, 1024 * 4, defaultScene, defaultSceneLoop);

  dprintf(2, "[STATS] Peak memory use: %d\n", gBufferPeakAllocatedSize);
  dprintf(2, "[STATS] Memory leak : %d\n", gBufferTotalAllocatedSize);
}
#include "objects.h"

void defaultScene(Scene* scene) {
  SceneInput inp = sceneInputHost(scene);

  int meshIdx     = 0;
  int materialIdx = 0;
  int objectIdx   = 0;
  int textureIdx  = 0;

  inp.meshes[meshIdx++] = meshPlain(make_float3(0, 1, 0));
  inp.meshes[meshIdx++] = meshPlain(make_float3(1, 0, 0));
  inp.meshes[meshIdx++] = meshPlain(make_float3(0, 0, 1));
  inp.meshes[meshIdx++] = meshPlain(make_float3(0, 1, 1));

  Material mat;
  inp.materials[materialIdx++] = mat = {};
  /*
  {
    .kd      = make_float3(0.5, 0.7, 0.8),
    .ks      = make_float3(0.2, 0.4, 0.5),
    .ka      = make_float3(0.1, 0.1, 0.1),
    .fresnel = 0.1,
    .ior     = 1.01}; */

  inp.materials[materialIdx++] = {
    .kd      = make_float3(0.8, 0.7, 0.2),
    .ks      = make_float3(0.2, 0.2, 0.2),
    .ka      = make_float3(0.1, 0.1, 0.1),
    .fresnel = 0.0,
    .ior     = 1.01};

  inp.materials[materialIdx++] = {
    .kd      = make_float3(10.8, 10.7, 10.2),
    .ks      = make_float3(0.2, 0.2, 0.2),
    .ka      = make_float3(0.1, 0.1, 0.1),
    .fresnel = 0.0,
    .ior     = 1.01};

  inp.materials[materialIdx++] = {
    .kd      = make_float3(0.01, 0.1, 0.2),
    .ks      = make_float3(0.8, 1.0, 1.0),
    .ka      = make_float3(0.1, 0.1, 0.1),
    .fresnel = 1.0,
    .ior     = 1.01};

  inp.objects[objectIdx++] = {.material = 0, .mesh = 1, .origin = make_float3(0, 1, 1)};
  inp.objects[objectIdx++] = {.material = 1, .mesh = 2, .origin = make_float3(-1, 1, 1)};
  inp.objects[objectIdx++] = {.material = 2, .mesh = 3, .origin = make_float3(-2, -1, 1)};
  inp.objects[objectIdx++] = {.material = 1, .mesh = 0, .origin = make_float3(2, 1, 1)};
  inp.objects[objectIdx++] = {.material = 3, .mesh = 0, .origin = make_float3(1, -1, 1)};

  inp.textures[textureIdx++] = textureCreate("assets/soil.png");

  scene->objectCount   = objectIdx;
  scene->materialCount = materialIdx;
  scene->meshCount     = meshIdx;
  scene->textureCount  = textureIdx;
}

int defaultSceneLoop(Object* objects, float t) {
  int objectIdx        = 0;
  objects[objectIdx++] = {.material = 0, .mesh = 1, .origin = make_float3(t, 1, 1)};
  objects[objectIdx++] = {.material = 1, .mesh = 2, .origin = make_float3(-1, 1, 1)};
  objects[objectIdx++] = {.material = 2, .mesh = 3, .origin = make_float3(-2, -1, 1)};
  objects[objectIdx++] = {.material = 1, .mesh = 0, .origin = make_float3(2, 1, 1)};
  objects[objectIdx++] = {.material = 3, .mesh = 0, .origin = make_float3(1, -1, 1)};
  return objectIdx;
}

#include <stdio.h>
#include <stb/stb_image_write.h>
#include "texture.h"
#include "util/buffer.h"
#include "objects.h"
#include "scene.h"



void defaultScene(Scene* scene);
void defaultSceneLoop(PushConstants* cn);

void programRun(const char* path, int width, int height, void(initScene)(Scene*), void(initSceneFrame)(PushConstants* cn)) {

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
    SceneInput sc = sceneInputHost(&scene);
    float      t  = 0;

    for (int i = 0; i < sceneDesc.framesInFlight; i++) {
      PushConstants* constants = &sc.constants[i];
      constants->frameTime     = t;
      initSceneFrame(constants);
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

Material createMaterial(Material mat) { return mat; }

/**
* Scene configuration
**/

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
  inp.materials[materialIdx++] = materialCreate(
    make_float3(0.5, 0.7, 0.8),
    make_float3(0.2, 0.4, 0.5),
    make_float3(0.1, 0.1, 0.1),
    0.1,
    1.01);

  inp.materials[materialIdx++] = materialCreate(
    make_float3(0.8, 0.7, 0.2),
    make_float3(0.2, 0.2, 0.2),
    make_float3(0.1, 0.1, 0.1),
    0.0,
    1.01);

  inp.materials[materialIdx++] = materialCreate(
    make_float3(10.8, 10.7, 10.2),
    make_float3(0.2, 0.2, 0.2),
    make_float3(0.1, 0.1, 0.1),
    0.0,
    1.01);

  inp.materials[materialIdx++] = materialCreate(
    make_float3(0.01, 0.1, 0.2),
    make_float3(0.8, 1.0, 1.0),
    make_float3(0.1, 0.1, 0.1),
    1.0,
    1.01);

  inp.textures[textureIdx++] = textureCreate("assets/soil.png");

  scene->materialCount = materialIdx;
  scene->meshCount     = meshIdx;
  scene->textureCount  = textureIdx;
}

void defaultSceneLoop(PushConstants* cn) {
  cn->uniforms.skyColor    = make_float3(0.2, 0.4, 0.9);
  cn->uniforms.groundColor = make_float3(0.2, 0.2, 0.2);
  cn->uniforms.orizonColor = make_float3(0.7, 0.8, 0.9);

  cn->camera.up        = make_float3(0, 1, 0);
  cn->camera.znear     = 0.1f;
  cn->camera.origin    = make_float3(0, 0, 0);
  cn->camera.direction = make_float3(0, 0, -1);

  int objectIdx            = 0;
  cn->objects[objectIdx++] = objectCreate(0, 1, make_float3(0, 1, 1));
  cn->objects[objectIdx++] = objectCreate(1, 2, make_float3(-1, 1, 1));
  cn->objects[objectIdx++] = objectCreate(2, 3, make_float3(-2, -1, 1));
  cn->objects[objectIdx++] = objectCreate(1, 0, make_float3(2, 1, 1));
  cn->objects[objectIdx++] = objectCreate(3, 0, make_float3(1, -1, 1));

  cn->objectCount = objectIdx;
}

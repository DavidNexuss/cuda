#include <stdio.h>
#include "texture.h"
#include "util/buffer.h"
#include "objects.h"
#include "scene.h"
#include "examples/examples.h"


void programRun(SceneDesc sceneDesc, const char* path, void(initScene)(Scene*), void(initSceneFrame)(PushConstants* cn), int cpu) {

  Scene scene = sceneCreate(sceneDesc);

  //Inits scene materials and meshes
  {
    initScene(&scene);
    if (!cpu) sceneUpload(&scene);
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

    if (!cpu) sceneUploadObjects(&scene);
  }

  if (!cpu) sceneRun(&scene);
  if (!cpu) sceneDownload(&scene);

  if (cpu) sceneRunCPU(&scene);
  for (int i = 0; i < sceneDesc.framesInFlight; i++) {
    sceneWriteFrame(&scene, path, i);
  }
  sceneDestroy(&scene);
}

SceneDesc defaultDesc() {

  SceneDesc sceneDesc;
  sceneDesc.maxMeshes           = 300;
  sceneDesc.maxObjects          = 400;
  sceneDesc.maxMaterials        = 300;
  sceneDesc.maxTextures         = 10;
  sceneDesc.maxVertexBuffer     = 10;
  sceneDesc.maxIndexBuffer      = 10;
  sceneDesc.frameBufferWidth    = 1024;
  sceneDesc.frameBufferHeight   = 1024;
  sceneDesc.numThreads          = 8;
  sceneDesc.iterationsPerThread = 4;
  sceneDesc.rayDepth            = 4;
  sceneDesc.framesInFlight      = 4;
  sceneDesc.frameDelta          = 0.1;
  sceneDesc.fWriteClamped       = 1;
  return sceneDesc;
}
void test1() {

  SceneDesc sceneDesc = defaultDesc();

  programRun(sceneDesc, "results/result_cpu.png", defaultScene, defaultSceneLoop, 1);
  programRun(sceneDesc, "results/result_gpu.png", defaultScene, defaultSceneLoop, 0);

  sceneDesc.frameBufferWidth *= 2;
  sceneDesc.frameBufferHeight *= 2;
  programRun(sceneDesc, "results/result_cpu_2.png", defaultScene, defaultSceneLoop, 1);
  programRun(sceneDesc, "results/result_gpu_2.png", defaultScene, defaultSceneLoop, 0);
}

void test2() {

  SceneDesc sceneDesc           = defaultDesc();
  sceneDesc.frameBufferWidth    = 1024 * 4;
  sceneDesc.frameBufferHeight   = 1024 * 4;
  sceneDesc.framesInFlight      = 1;
  sceneDesc.fWriteClamped       = 1;
  sceneDesc.iterationsPerThread = 32;
  sceneDesc.numThreads          = 16;
  programRun(sceneDesc, "results/result_gpu_test2.png", scene2, scene2Loop, 0);
}

int main(int argc, char** argv) {
  test2();
  bufferDebugStats();
}

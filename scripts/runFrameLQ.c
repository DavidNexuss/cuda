#include <scene.h>

static SceneDesc defaultDesc() {

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
  sceneDesc.framesInFlight      = 1;
  sceneDesc.frameDelta          = 0.1;
  sceneDesc.fWriteClamped       = 1;
  return sceneDesc;
}


void traceInit(Scene* scene);
void traceLoop(PushConstants* cn);
int  main(int argc, char** argv) {
  SceneDesc sceneDesc           = defaultDesc();
  sceneDesc.frameBufferWidth    = 1280 / 2;
  sceneDesc.frameBufferHeight   = 720 / 2;
  sceneDesc.framesInFlight      = 1;
  sceneDesc.fWriteClamped       = 1;
  sceneDesc.iterationsPerThread = 1;
  sceneDesc.numThreads          = 32;
  sceneRunSuite(sceneDesc, "results/frame.png", traceInit, traceLoop, 0);
  bufferDebugStats();
}

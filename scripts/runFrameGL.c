#include <scene.h>
#include "../backends/backend.h"

static SceneDesc defaultDesc() {

  SceneDesc sceneDesc;
  sceneDesc.maxMeshes           = 300;
  sceneDesc.maxObjects          = 400;
  sceneDesc.maxMaterials        = 400;
  sceneDesc.maxTextures         = 400;
  sceneDesc.maxVertexBuffer     = 400;
  sceneDesc.maxIndexBuffer      = 400;
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

int main(int argc, char** argv) {
  SceneDesc sceneDesc           = defaultDesc();
  sceneDesc.frameBufferWidth    = 1280;
  sceneDesc.frameBufferHeight   = 720;
  sceneDesc.framesInFlight      = 1;
  sceneDesc.fWriteClamped       = 1;
  sceneDesc.iterationsPerThread = 16;
  sceneDesc.numThreads          = 32;

  RendererDesc rendererDesc;
  rendererDesc.width  = 1080;
  rendererDesc.height = 720;

  Scene scene = sceneCreate(sceneDesc);
  traceInit(&scene);

  SceneInput in          = sceneInputHost(&scene);
  int        objectCount = in.constants->objectCount;

  Renderer* renderer = rendererCreate(rendererDesc);
  rendererUpload(renderer, &scene);
  do {
    SceneInput in = sceneInputHost(&scene);
    for (int i = 0; i < scene.desc.framesInFlight; i++) {
      (in.constants + i)->objectCount = objectCount;
      traceLoop(in.constants + i);
    }

    rendererUploadObjects(renderer, &scene);
    rendererDraw(renderer, &scene);

  } while (rendererPollEvents(renderer));

  rendererDestoy(renderer);
  // sceneDestroy(&scene);
  bufferDebugStats();
}

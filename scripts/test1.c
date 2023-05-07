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

void traceInit(Scene* scene) {
  SceneInput inp = sceneInputHost(scene);

  int meshIdx     = 0;
  int materialIdx = 0;
  int textureIdx  = 0;

  inp.meshes[meshIdx++] = meshPlain(make_float3(0, 1, 0));
  inp.meshes[meshIdx++] = meshPlain(make_float3(1, 1, 0));

  Material mat;
  inp.materials[materialIdx++] = materialCreate(
    make_float3(0.5, 0.7, 0.8),
    make_float3(0.2, 0.4, 0.5),
    make_float3(0.1, 0.1, 0.1),
    0.1,
    1.01);
  inp.materials[0].diffuseTexture = 0;

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

  inp.textures[textureIdx++] = textureCreate("assets/checker.png");
  inp.textures[textureIdx++] = textureCreate("assets/equi2.png");
  inp.textures[textureIdx++] = textureCreate("assets/stone.jpg");

  scene->materialCount = materialIdx;
  scene->meshCount     = meshIdx;
  scene->textureCount  = textureIdx;
}

void traceLoop(PushConstants* cn) {
  cn->uniforms.skyColor    = make_float3(0.2, 0.4, 0.9);
  cn->uniforms.groundColor = make_float3(0.2, 0.2, 0.2);
  cn->uniforms.orizonColor = make_float3(0.7, 0.8, 0.9);
  cn->uniforms.skyTexture  = 1;
  cn->camera.znear     = 0.1f;

  cn->camera.up        = make_float3(0, 1, 0);
  cn->camera.origin    = make_float3(0, 1, 0);
  cn->camera.direction = make_float3(0, 0, -1);
  cn->camera.crossed   = make_float3(1, 0, 0);

  int objectIdx            = 0;
  cn->objects[objectIdx++] = objectCreate(0, 0, make_float3(0, -1, 1));

  cn->objectCount = objectIdx;
}

#ifdef IMPL
int main(int argc, char** argv) {
  SceneDesc sceneDesc           = defaultDesc();
  sceneDesc.frameBufferWidth    = 1920;
  sceneDesc.frameBufferHeight   = 1080;
  sceneDesc.framesInFlight      = 1;
  sceneDesc.fWriteClamped       = 1;
  sceneDesc.iterationsPerThread = 2;
  sceneDesc.numThreads          = 32 * 4;

  sceneRunSuite(sceneDesc, "results/result_gpu_test2.png", traceInit, traceLoop, 0);
  bufferDebugStats();
}
#endif

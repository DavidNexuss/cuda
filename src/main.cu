#include <stdio.h>
#include <stb/stb_image_write.h>
#include "util/buffer.h"
#include "util/debug.h"
#include "trace.h"

//SCENE
typedef struct {
  int maxObjects;
  int maxMaterials;
  int maxMeshes;
  int frameBufferWidth;
  int frameBufferHeight;
  int iterationCount;

} SceneDesc;

typedef struct {
  //Input buffer objects
  Buffer meshes;
  Buffer objects;
  Buffer materials;

  //Output buffer objects
  Buffer framebuffer;

  //Push constants
  Camera camera;
  int    objectCount;

  //Scene configuration
  SceneDesc desc;

} Scene;

typedef struct {
  Object*   objects;
  Material* materials;
  Mesh*     meshes;
} SceneInput;

Scene sceneCreate(SceneDesc desc) {
  Scene scene;
  scene.desc        = desc;
  scene.meshes      = bufferCreate(sizeof(Mesh) * desc.maxMeshes);
  scene.objects     = bufferCreate(sizeof(Object) * desc.maxObjects);
  scene.materials   = bufferCreate(sizeof(Material) * desc.maxMaterials);
  scene.framebuffer = bufferCreate(3 * sizeof(float) * desc.frameBufferWidth * desc.frameBufferHeight);
  return scene;
}

SceneInput sceneInputHost(Scene* scene, int objectCount) {
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

//Execute path tracing on the scene with the given parameters
__global__ void pathTracingKernel(SceneInput sceneInput, Camera cam, int objectCount, int width, int height, float* fbo, int iterationsPerThread) {
  float u = blockIdx.x / float(width);
  float v = blockIdx.y / float(height);

  int pixelIdx = (blockIdx.x * width + blockIdx.y) * 3;

  //Default uv gradient test
  fbo[pixelIdx]     = u;
  fbo[pixelIdx + 1] = v;
  fbo[pixelIdx + 2] = threadIdx.x / float(blockDim.x);
}

void sceneRun(Scene* scene) {
  dim3 numBlocks           = dim3(scene->desc.frameBufferWidth, scene->desc.frameBufferHeight, 1);
  int  numThreads          = scene->desc.iterationCount;
  int  iterationsPerThread = 1;
  pathTracingKernel<<<numBlocks, numThreads>>>({
                                                 (Object*)scene->objects.D,
                                                 (Material*)scene->materials.H,
                                                 (Mesh*)scene->meshes.H,
                                               },
                                               scene->camera, scene->objectCount, scene->desc.frameBufferWidth, scene->desc.frameBufferHeight, (float*)scene->framebuffer.D, iterationsPerThread);
}

void programRun(const char* path, int width, int height) {

  SceneDesc sceneDesc         = {};
  sceneDesc.maxMeshes         = 300;
  sceneDesc.maxObjects        = 400;
  sceneDesc.maxMaterials      = 300;
  sceneDesc.frameBufferWidth  = 1024;
  sceneDesc.frameBufferHeight = 1024;
  sceneDesc.iterationCount    = 512;

  Scene scene = sceneCreate(sceneDesc);

  {
    auto init = sceneInputHost(&scene, 2);
    sceneUpload(&scene);
  }

  sceneRun(&scene);
  sceneDownload(&scene);

  stbi_write_hdr(path, width, height, 3, (const float*)scene.framebuffer.H);
  sceneDestroy(&scene);
}

int main(int argc, char** argv) {

  programRun("result.hdr", 1024, 1024);
  LOG("[STATS] Peak memory use: %d\n", gBufferPeakAllocatedSize);
  LOG("[STATS] Memory leak : %d\n", gBufferTotalAllocatedSize);
}

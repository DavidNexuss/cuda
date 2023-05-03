#include <stdio.h>
#include <stb/stb_image_write.h>
#include "util/buffer.h"
#include "util/debug.h"
#include "trace.h"

/*
 This struct defines all configurable parameters for class scene
*/
typedef struct {
  int maxObjects;
  int maxMaterials;
  int maxMeshes;
  int frameBufferWidth;
  int frameBufferHeight;
  int iterationCount;
  int rayDepth;

} SceneDesc;

/**
  This class holds memory and handles for:
    - meshes
    - objects
    - materials
    - the framebuffer
*/
typedef struct _Scene {
  //Input buffer objects
  Buffer meshes;
  Buffer objects;
  Buffer materials;

  //Output buffer objects
  Buffer framebuffer;

  //Push constants
  Camera camera;
  int    objectCount   = -1;
  int    materialCount = -1;
  int    meshCount     = -1;

  //Scene configuration
  SceneDesc desc;

} Scene;

/* bag of pointers */
typedef struct {
  Object*   objects;
  Material* materials;
  Mesh*     meshes;
} SceneInput;

/* creates a scene using scenedesc */
Scene sceneCreate(SceneDesc desc) {
  Scene scene;
  scene.desc        = desc;
  scene.meshes      = bufferCreate(sizeof(Mesh) * desc.maxMeshes);
  scene.objects     = bufferCreate(sizeof(Object) * desc.maxObjects);
  scene.materials   = bufferCreate(sizeof(Material) * desc.maxMaterials);
  scene.framebuffer = bufferCreate(3 * sizeof(float) * desc.frameBufferWidth * desc.frameBufferHeight);
  return scene;
}

/* returns host bag of pointers */
SceneInput sceneInputHost(Scene* scene) {
  scene->objectCount = 0;
  return {
    (Object*)scene->objects.H,
    (Material*)scene->materials.H,
    (Mesh*)scene->meshes.H};
}

/* destroys scene and releases memory */
void sceneDestroy(Scene* scene) {
  bufferDestroy(&scene->meshes);
  bufferDestroy(&scene->objects);
  bufferDestroy(&scene->materials);
  bufferDestroy(&scene->framebuffer);
}

/* uploads scene */
void sceneUpload(Scene* scene) {
  bufferUpload(&scene->materials, scene->materialCount * sizeof(Material));
  bufferUpload(&scene->objects, scene->objectCount * sizeof(Object));
  bufferUpload(&scene->meshes, scene->meshCount * sizeof(Mesh));
}

/* downloads scene */
void sceneDownload(Scene* scene) {
  bufferDownload(&scene->framebuffer);
}

//Execute path tracing on the scene with the given parameters
__global__ void pathTracingKernel(SceneInput sceneInput, Camera cam, int objectCount, int width, int height, float* fbo, int iterationsPerThread, int maxDepth) {
  float u = blockIdx.x / float(width);
  float v = blockIdx.y / float(height);

  int pixelIdx = (blockIdx.x * width + blockIdx.y) * 3;
  int thread   = threadIdx.x;

  extern __shared__ float3 result[];

  float3 sro = cam.origin;
  float3 srd = make_float3(u * 2 - 1, v * 2 - 1, 1);

  //Perform path tracing using rd and ro

#if 0
  float3 threadResult;
  for (int i = 0; i < iterationsPerThread; i++) {
    float3 partialResult = make_float3(0, 0, 0);

    float3 ro = sro;
    float3 rd = srd;

    for (int d = 0; d < maxDepth; d++) {
    }
  }
#endif 

  //Default uv gradient test
  fbo[pixelIdx]     = u;
  fbo[pixelIdx + 1] = v;
  fbo[pixelIdx + 2] = threadIdx.x / float(blockDim.x);
}

void sceneRun(Scene* scene) {
  dim3 numBlocks           = dim3(scene->desc.frameBufferWidth, scene->desc.frameBufferHeight, 1);
  int  numThreads          = scene->desc.iterationCount;
  int  iterationsPerThread = 1;
  pathTracingKernel<<<numBlocks, numThreads, sizeof(float) * 3 * numThreads>>>({(Object*)scene->objects.D, (Material*)scene->materials.H, (Mesh*)scene->meshes.H},
                                                                               scene->camera, scene->objectCount, scene->desc.frameBufferWidth, scene->desc.frameBufferHeight, (float*)scene->framebuffer.D, iterationsPerThread, scene->desc.rayDepth);
}

void defaultScene(Scene* scene);
void programRun(const char* path, int width, int height, void(initSceneFunction)(Scene*)) {

  SceneDesc sceneDesc         = {};
  sceneDesc.maxMeshes         = 300;
  sceneDesc.maxObjects        = 400;
  sceneDesc.maxMaterials      = 300;
  sceneDesc.frameBufferWidth  = width;
  sceneDesc.frameBufferHeight = height;
  sceneDesc.iterationCount    = 512;
  sceneDesc.rayDepth          = 4;

  Scene scene = sceneCreate(sceneDesc);
  initSceneFunction(&scene);
  sceneUpload(&scene);
  sceneRun(&scene);
  sceneDownload(&scene);
  stbi_write_hdr(path, width, height, 3, (const float*)scene.framebuffer.H);
  sceneDestroy(&scene);
}

int main(int argc, char** argv) {

  programRun("result.hdr", 1024, 1024, defaultScene);
  programRun("result2.hdr", 1024 * 2, 1024 * 2, defaultScene);

  LOG("[STATS] Peak memory use: %d\n", gBufferPeakAllocatedSize);
  LOG("[STATS] Memory leak : %d\n", gBufferTotalAllocatedSize);
}

void defaultScene(Scene* scene) {
  SceneInput inp = sceneInputHost(scene);

  int meshIdx     = 0;
  int materialIdx = 0;
  int objectIdx   = 0;

  inp.meshes[meshIdx++] = meshPlain(make_float3(0, 1, 0));
  inp.meshes[meshIdx++] = meshPlain(make_float3(1, 0, 0));
  inp.meshes[meshIdx++] = meshPlain(make_float3(0, 0, 1));
  inp.meshes[meshIdx++] = meshPlain(make_float3(0, 1, 1));

  inp.materials[materialIdx++] = {
    .kd      = vec3(0.5, 0.7, 0.8),
    .ks      = vec3(0.2, 0.4, 0.5),
    .ka      = vec3(0.1, 0.1, 0.1),
    .fresnel = 0.1,
    .ior     = 1.01};

  inp.materials[materialIdx++] = {
    .kd      = vec3(0.8, 0.7, 0.2),
    .ks      = vec3(0.2, 0.2, 0.2),
    .ka      = vec3(0.1, 0.1, 0.1),
    .fresnel = 0.0,
    .ior     = 1.01};

  inp.materials[materialIdx++] = {
    .kd      = vec3(10.8, 10.7, 10.2),
    .ks      = vec3(0.2, 0.2, 0.2),
    .ka      = vec3(0.1, 0.1, 0.1),
    .fresnel = 0.0,
    .ior     = 1.01};

  inp.materials[materialIdx++] = {
    .kd      = vec3(0.01, 0.1, 0.2),
    .ks      = vec3(0.8, 1.0, 1.0),
    .ka      = vec3(0.1, 0.1, 0.1),
    .fresnel = 1.0,
    .ior     = 1.01};

  inp.objects[objectIdx++] = {.material = 0, .mesh = 1, .origin = vec3(0, 1, 1)};
  inp.objects[objectIdx++] = {.material = 1, .mesh = 2, .origin = vec3(-1, 1, 1)};
  inp.objects[objectIdx++] = {.material = 2, .mesh = 3, .origin = vec3(-2, -1, 1)};
  inp.objects[objectIdx++] = {.material = 1, .mesh = 0, .origin = vec3(2, 1, 1)};
  inp.objects[objectIdx++] = {.material = 3, .mesh = 0, .origin = vec3(1, -1, 1)};

  scene->objectCount   = objectIdx;
  scene->materialCount = materialIdx;
  scene->meshCount     = meshIdx;
}

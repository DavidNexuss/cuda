#include <stdio.h>
#include <stb/stb_image_write.h>
#include "util/buffer.h"
#include "util/debug.h"
#include "trace.h"

//SCENE
typedef struct {
  int   maxObjects;
  int   maxMaterials;
  int   maxMeshes;
  int   frameBufferWidth;
  int   frameBufferHeight;
  int   numThreads;
  int   iterationsPerThread;
  int   rayDepth;
  int   framesInFlight;
  float frameDelta;

} SceneDesc;

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
  scene.framebuffer = bufferCreate(3 * sizeof(float) * desc.frameBufferWidth * desc.frameBufferHeight * desc.framesInFlight);
  return scene;
}

SceneInput sceneInputHost(Scene* scene) {
  scene->objectCount = 0;
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
  bufferUpload(&scene->materials, scene->materialCount * sizeof(Material));
  bufferUpload(&scene->meshes, scene->meshCount * sizeof(Mesh));
}

void sceneUploadObjects(Scene* scene) {
  bufferUpload(&scene->objects, scene->objectCount * sizeof(Object) * scene->desc.framesInFlight);
}

void sceneDownload(Scene* scene) {
  bufferDownload(&scene->framebuffer);
}

float* sceneGetFrame(Scene* scene, int index) {
  return (float*)&scene->framebuffer.H[index * 3 * sizeof(float) * scene->desc.frameBufferWidth * scene->desc.frameBufferHeight];
}

void sceneWriteFrame(Scene* scene, const char* path, int index) {
  stbi_write_hdr(path, scene->desc.frameBufferWidth, scene->desc.frameBufferHeight, 3, sceneGetFrame(scene, index));
}

//Execute path tracing on the scene with the given parameters
__global__ void pathTracingKernel(SceneInput sceneInput, Camera cam, int objectCount, int width, int height, float* fbo_mat, int iterationsPerThread, int maxDepth) {
  float u = blockIdx.x / float(width);
  float v = blockIdx.y / float(height);

  int pixelIdx = (blockIdx.x * width + blockIdx.y) * 3;
  int thread   = threadIdx.x;

  extern __shared__ float3 result[];

  float* fbo = fbo_mat + blockIdx.z * width * height * 3;

  float3  sro     = cam.origin;
  float3  srd     = make_float3(u * 2 - 1, v * 2 - 1, 1);
  Object* objects = sceneInput.objects + blockIdx.z * objectCount;


  //Perform path tracing using rd and ro
  /*
  float3 threadResult;
  for (int i = 0; i < iterationsPerThread; i++) {
    float3 partialResult = make_float3(0, 0, 0);

    float3 ro = sro;
    float3 rd = srd;

    for (int d = 0; d < maxDepth; d++) {
    }
  } */

  //Default uv gradient test
  fbo[pixelIdx]     = u;
  fbo[pixelIdx + 1] = v;
  fbo[pixelIdx + 2] = threadIdx.x / float(blockDim.x);
}

void sceneRun(Scene* scene) {
  dim3 numBlocks           = dim3(scene->desc.frameBufferWidth, scene->desc.frameBufferHeight, scene->desc.framesInFlight);
  int  numThreads          = scene->desc.numThreads;
  int  iterationsPerThread = scene->desc.iterationsPerThread;
  LOG("Running path tracing kernel [%d, %d, %d] with %d threads, iterations per thread: %d\n", numBlocks.x, numBlocks.y, numBlocks.z, numThreads, iterationsPerThread);

  pathTracingKernel<<<numBlocks, numThreads, sizeof(float) * 3 * numThreads>>>({(Object*)scene->objects.D, (Material*)scene->materials.H, (Mesh*)scene->meshes.H},
                                                                               scene->camera, scene->objectCount, scene->desc.frameBufferWidth, scene->desc.frameBufferHeight, (float*)scene->framebuffer.D, iterationsPerThread, scene->desc.rayDepth);
}

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
  sceneDesc.framesInFlight      = 8;
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

int defaultSceneLoop(Object* objects, float t) {
  int objectIdx        = 0;
  objects[objectIdx++] = {.material = 0, .mesh = 1, .origin = vec3(t, 1, 1)};
  objects[objectIdx++] = {.material = 1, .mesh = 2, .origin = vec3(-1, 1, 1)};
  objects[objectIdx++] = {.material = 2, .mesh = 3, .origin = vec3(-2, -1, 1)};
  objects[objectIdx++] = {.material = 1, .mesh = 0, .origin = vec3(2, 1, 1)};
  objects[objectIdx++] = {.material = 3, .mesh = 0, .origin = vec3(1, -1, 1)};
  return objectIdx;
}

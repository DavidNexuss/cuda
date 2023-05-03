#include <stdio.h>
#include <stb/stb_image_write.h>
#include "util/buffer.h"
#include "trace.h"

typedef struct { 
  int width;
  int height;
  int channels;
  Buffer data;
} Texture;


Texture textureCreate(const char* texturePath) { 
  Texture text;
  void* mem = stbi_load(texturePath, &text.width, &text.height, &text.channels);
  text.data = bufferCreate(text.width * text.height * text.channels);
  memcpy(text.data.H, mem, text.data.allocatedSize);
  stbi_free(mem);
  return text;
}

typedef struct { 
  dim3 skyColor;
  dim3 orizonColor;
  dim3 groundColor;
} RenderingUniforms;


/*
 This struct defines all configurable parameters for class scene
*/
typedef struct {
  int maxObjects;
  int maxMaterials;
  int maxMeshes;
  int maxTextures;
  int frameBufferWidth;
  int frameBufferHeight;
  int iterationCount;
  int rayDepth;
  RenderingUniforms uniforms;

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
  Buffer textures;

  //Output buffer objects
  Buffer framebuffer;

  //Push constants
  Camera camera;
  int    objectCount   = -1;
  int    materialCount = -1;
  int    meshCount     = -1;
  int    textureCount = -1;

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
  scene.textures    = bufferCreate(sizeof(Texture) * desc.maxTextures);
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
    (Mesh*)scene->meshes.H,
    (Texture*)scene->textures.H};
}

/* destroys scene and releases memory */
void sceneDestroy(Scene* scene) {
  bufferDestroy(&scene->meshes);
  bufferDestroy(&scene->objects);
  bufferDestroy(&scene->materials);
  bufferDestroy(&scene->framebuffer);
  
  for(int i = 0; i < scene->textureCount; i++) { 
    bufferDestroy(&scene->textures[i].data);
  }
  bufferDestroy(&scene->textures);
}

/* uploads scene */
void sceneUpload(Scene* scene) {
  bufferUpload(&scene->materials, scene->materialCount * sizeof(Material));
  bufferUpload(&scene->objects, scene->objectCount * sizeof(Object));
  bufferUpload(&scene->meshes, scene->meshCount * sizeof(Mesh));
  bufferUpload(&scene->textures, scene->texureCount * sizeof(Texture));

  for(int i = 0; i < scene->textureCount; i++) { 
    bufferUpload(&scene->textures[i].data, scene->textures[i].allocatedSize);
  }
}

/* downloads scene */
void sceneDownload(Scene* scene) {
  bufferDownload(&scene->framebuffer);
}

//Execute path tracing on the scene with the given parameters
__global__ void pathTracingKernel(SceneInput sceneInput, Camera cam, int objectCount, int width, int height, float* fbo, int iterationsPerThread, int maxDepth, RenderingUniforms uniforms) {
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

  float3 result = 

  fbo[pixelIdx]     = result.x;
  fbo[pixelIdx + 1] = result.y;
  fbo[pixelIdx + 2] = result.z;
#if 0
  //Default uv gradient test
  fbo[pixelIdx]     = u;
  fbo[pixelIdx + 1] = v;
  fbo[pixelIdx + 2] = threadIdx.x / float(blockDim.x);
#endif
}

void sceneRun(Scene* scene) {
  dim3 numBlocks           = dim3(scene->desc.frameBufferWidth, scene->desc.frameBufferHeight, 1);
  int  numThreads          = scene->desc.iterationCount;
  int  iterationsPerThread = 1;
  pathTracingKernel<<<numBlocks, numThreads, sizeof(float) * 3 * numThreads>>>(
									       {(Object*)scene->objects.D, (Material*)scene->materials.D, (Mesh*)scene->meshes.D},
                                                                               scene->camera, scene->objectCount, scene->desc.frameBufferWidth, scene->desc.frameBufferHeight, 
									       (float*)scene->framebuffer.D, iterationsPerThread, scene->desc.rayDepth, scene->desc.uniforms);
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
  programRun("result3.hdr", 1024 * 4, 1024 * 4, defaultScene);

  dprintf(2, "[STATS] Peak memory use: %d\n", gBufferPeakAllocatedSize);
  dprintf(2, "[STATS] Memory leak : %d\n", gBufferTotalAllocatedSize);
}

void defaultScene(Scene* scene) {
  SceneInput inp = sceneInputHost(scene);

  int meshIdx     = 0;
  int materialIdx = 0;
  int objectIdx   = 0;
  int textureIdx = 0;

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
scene->textureCount = textureIdx;
}

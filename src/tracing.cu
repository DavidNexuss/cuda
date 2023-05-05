#include <stdio.h>

extern "C" {
#include "scene.h"
}

__device__ float3 prod(float3 a, float3 b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__device__ float3 prodScalar(float3 a, float t) {
  return prod(a, make_float3(t, t, t));
}
__device__ float3 sub(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 sum(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

//This are cuda versions of basic linear algebra functionality
__device__ float3 lReflect(float3 rd, float3 normal);
__device__ float3 lRefract(float3 rd, float3 normal, float ior);
__device__ float3 lNormalize(float3 v);
__device__ float  lLen2(float3 a);
__device__ float  lLen(float3 a);

//Returns origin + direction * distance
__device__ float3 lAdvance(float3 origin, float3 direction, float distance);

//Retuns a normalized random direction
__device__ float3 lRandomDirection();

//Retuns a normalized random direction in a hemisphere from normalvector
__device__ float3 lRrandomDirectionHemisphere(float3 normalvector);

//Returns sky color
__device__ float3 lClearColorBackground(float3 rd, float3 ground, float3 orizon, float3 sky) {
  float t = rd.y;
  return sub(prodScalar(sky, t), prodScalar(ground, -t));
}

//Signed distance field functions combined with direction optimisation whenever possible
__device__ int sdfHitSphere(float3 ro, float3 rd, float radius, float* delta, float3* normal);
__device__ int sdfHitPlane(float3 ro, float3 rd, float3 normal, float* delta, float* normalDir);

__device__ float3 sampleTexture(dim3* rgb, float2 uv);

__global__ void pathTracingKernel(int width, int height, float* fbo_mat, int iterationsPerThread, int maxDepth, SceneInput input) {
  Camera* cam     = &(input.constants + blockIdx.z)->camera;
  Object* objects = (input.constants + blockIdx.z)->objects;

  float u = blockIdx.x / float(width);
  float v = blockIdx.y / float(height);

  int pixelIdx = (blockIdx.x * width + blockIdx.y) * 3;
  int thread   = threadIdx.x;

  extern __shared__ float3 result[];

  float* fbo = fbo_mat + blockIdx.z * width * height * 3;

  float3 sro = cam->origin;
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


#if 0
  fbo[pixelIdx]   = result.x;
  fbo[pixelIdx + 1] = result.y;
  fbo[pixelIdx + 2] = result.z;
#endif
#if 1
  //Default uv gradient test
  fbo[pixelIdx]     = u;
  fbo[pixelIdx + 1] = v;
  fbo[pixelIdx + 2] = threadIdx.x / float(blockDim.x);
#endif
}

extern "C" {
static int jobIdCounter = 0;
void sceneRun(Scene* scene) {
  dim3 numBlocks           = dim3(scene->desc.frameBufferWidth, scene->desc.frameBufferHeight, scene->desc.framesInFlight);
  int  numThreads          = scene->desc.numThreads;
  int  iterationsPerThread = scene->desc.iterationsPerThread;
  int jobId = jobIdCounter;
  dprintf(2, "[%d] Running path tracing kernel [%d, %d, %d] with %d threads, iterations per thread: %d\n", jobId, numBlocks.x, numBlocks.y, numBlocks.z, numThreads, iterationsPerThread);

  pathTracingKernel<<<numBlocks, numThreads, sizeof(float) * 3 * numThreads>>>(
    scene->desc.frameBufferWidth,
    scene->desc.frameBufferHeight, (float*)scene->framebuffer.D, iterationsPerThread, scene->desc.rayDepth, sceneInputDevice(scene));
  dprintf(2, "[%d] done\n", jobId);
  jobIdCounter++;
}
}

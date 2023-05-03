#include "tracing.h"
#include <stdio.h>

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
#if 0
  //Default uv gradient test
  fbo[pixelIdx]     = u;
  fbo[pixelIdx + 1] = v;
  fbo[pixelIdx + 2] = threadIdx.x / float(blockDim.x);
#endif
}

void sceneRun(Scene* scene) {
  dim3 numBlocks           = dim3(scene->desc.frameBufferWidth, scene->desc.frameBufferHeight, scene->desc.framesInFlight);
  int  numThreads          = scene->desc.numThreads;
  int  iterationsPerThread = scene->desc.iterationsPerThread;
  dprintf(2, "Running path tracing kernel [%d, %d, %d] with %d threads, iterations per thread: %d\n", numBlocks.x, numBlocks.y, numBlocks.z, numThreads, iterationsPerThread);

  pathTracingKernel<<<numBlocks, numThreads, sizeof(float) * 3 * numThreads>>>(
    scene->desc.frameBufferWidth,
    scene->desc.frameBufferHeight, (float*)scene->framebuffer.D, iterationsPerThread, scene->desc.rayDepth, sceneInputDevice(scene));
}

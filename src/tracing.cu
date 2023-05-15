#include <stdio.h>
#include "cuda/cutil_math.h"

extern "C" {
#include "../include/scene.h"
}

#define HEAD __host__ __device__

HEAD float3 prod(float3 a, float3 b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
HEAD float3 prodScalar(float3 a, float t) {
  return prod(a, make_float3(t, t, t));
}
HEAD float3 sub(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
HEAD float3 sum(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

//This are cuda versions of basic linear algebra functionality
HEAD float3 lReflect(float3 rd, float3 normal) {
  float s = dot(rd, normal);
  return make_float3(rd.x - 2 * s * normal.x,
                     rd.y - 2 * s * normal.y,
                     rd.z - 2 * s * normal.z);
}
HEAD float3 lRefract(float3 rd, float3 normal, float ior);
HEAD float  lLen2(float3 a) {
  return a.x * a.x + a.y * a.y + a.z * a.z;
}
HEAD float lLen(float3 a) {
  return sqrt(lLen2(a));
}

HEAD float3 lNormalize(float3 v) {
  float len = lLen(v);
  return make_float3(v.x / len, v.y / len, v.z / len);
}

HEAD unsigned int lHash(unsigned int x) {
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = (x >> 16) ^ x;
  return x;
}

//Returns origin + direction * distance
HEAD float3 lAdvance(float3 origin, float3 direction, float distance) {
  return make_float3(origin.x + direction.x * distance, origin.y + direction.y * distance, origin.z + direction.z * distance);
}

//Retuns a normalized random direction

//Retuns a normalized random direction in a hemisphere from normalvector
HEAD float3 lRrandomDirectionHemisphere(float3 normalvector);

//Returns sky color
HEAD float3 lClearColorBackground(float3 rd, float3 ground, float3 orizon, float3 sky) {
  float t = rd.y;
  return sub(prodScalar(sky, t), prodScalar(ground, -t));
}

HEAD float3 applyMatrix(float3 v, float3 x, float3 y, float3 z) {
  return make_float3(
    v.x * x.x + v.y * y.x + v.z * z.x,
    v.x * x.y + v.y * y.y + v.z * z.y,
    v.x * x.z + v.y * y.z + v.z * z.z);
}
HEAD float3 rotateVector(float3 rd, float3 y) {
  float3 x = make_float3(1, 0, 0);
  float3 z = cross(x, y);
  return applyMatrix(rd, x, y, z);
}
//Signed distance field functions combined with direction optimisation whenever possible
HEAD int sdfHitSphere(float3 ro, float3 rd, float radius, float* delta, float3* normal) {
  float3 oc           = ro;
  float  a            = dot(rd, rd);
  float  b            = 2.0 * dot(oc, rd);
  float  c            = dot(oc, oc) - radius * radius;
  float  discriminant = b * b - 4 * a * c;
  if (discriminant <= 0.0) return 0;

  *delta    = (-b - sqrt(discriminant)) / (2 * a);
  *normal   = lNormalize((ro + rd * *delta));
  normal->x = -normal->x;
  normal->y = -normal->y;
  normal->z = -normal->z;
  return 1;
}

HEAD int sdfHitPlane(float3 ro, float3 rd, float* delta) {
  if (rd.y > 0.0f) return 0;
  *delta = ro.y / rd.y;
  return 1;
}

HEAD float lRandom(int magic) { return fracf(float(magic) * 0.0001) * 2.0f - 1.0f; }

HEAD float3 lRandomDirection(int magic) { return lNormalize(make_float3(lRandom(magic), lRandom(magic * 43), lRandom(magic * 51))); }
HEAD float3 sampleTexture(Texture* text, float2 uv) {
  uv                 = fracf(uv);
  unsigned char* rgb = (unsigned char*)text->data;
  int            x   = uv.x * text->width;
  int            y   = uv.y * text->height;

  int i = (y * text->width + x) * 3;
  return make_float3(rgb[i] / float(255.0f), rgb[i + 1] / float(255.0f), rgb[i + 2] / float(255.0f));
}

HEAD float3 sampleEnvMap(Texture* text, float3 rd) {
  float  x      = 1.5 * atan2(rd.z, rd.x) / (2 * M_PI);
  float  y      = 0.5 + 2.0 * atan2(rd.y, sqrt(rd.x * rd.x + rd.z * rd.z)) / (2 * M_PI);
  float3 result = sampleTexture(text, make_float2(x, 1 - y));
  return prodScalar(prod(result, result), 2.5);
}
#define FLT_MAX 3.402823466e+38F /* max value */
HEAD float3 pathTracing(int width, int height, int iterationsPerThread, int maxDepth, SceneInput input, int x, int y, int frame, int magic) {
  float  ra = float(width) / float(height);
  float2 uv = make_float2((2 * x / float(width)) - 1, (2 * (height - y) / float(height)) - 1);
  uv.x *= ra;

  PushConstants* constants  = input.constants + frame;
  Texture*       skyTexture = &input.textures[constants->uniforms.skyTexture];
  Mesh*          meshes     = input.meshes;

  float3 rd = make_float3(uv.x, uv.y, 1);
  float3 ro = constants->camera.origin;
  //DOF
  magic = lHash(magic);
  rd.x  = rd.x + lRandom(lHash(magic)) * 0.002;
  rd.y  = rd.y + lRandom(lHash(magic + 71)) * 0.002;
  rd.z  = rd.z + lRandom(lHash(magic + 45)) * 0.002;

  float3 rdProjected = make_float3(
    constants->camera.crossed.x * rd.x + constants->camera.up.x * rd.y + constants->camera.direction.x * rd.z,
    constants->camera.crossed.y * rd.x + constants->camera.up.y * rd.y + constants->camera.direction.y * rd.z,
    constants->camera.crossed.z * rd.x + constants->camera.up.z * rd.y + constants->camera.direction.z * rd.z);

  rd = lNormalize(rdProjected);

  float3 currentColor = make_float3(1, 1, 1);
  for (int d = 0; d < 1; d++) {
    float3 hitNormal;
    float  delta           = FLT_MAX;
    int    collisionObject = -1;
    for (int i = 0; i < constants->objectCount; i++) {
      Object* obj = &constants->objects[i];
      float   partialDelta;
      float3  partialNormal;
      int     hitStatus = 0;
      Mesh*   mesh      = &meshes[obj->mesh];
      switch (mesh->type) {
        case PLAIN:
          partialNormal = mesh->tPlain.normal;
          hitStatus     = sdfHitPlane(ro, rotateVector(rd, partialNormal), &partialDelta);
          break;
        case SPHERE:
          hitStatus = sdfHitSphere(ro, rd, 0.8, &partialDelta, &partialNormal);
          break;
        case MESH:
          break;
        case CUSTOM:
          hitStatus = (*mesh->tCustom.sdf)(ro, rd, &partialDelta);
          break;
      }

      if (hitStatus == 1) {
        if (partialDelta < delta) {
          delta           = partialDelta;
          hitNormal       = partialNormal;
          collisionObject = i;
        }
      }
    }

    if (collisionObject == -1) {
      return prod(currentColor, sampleEnvMap(skyTexture, rd) * 4.0);
    }

    float fresnel = abs(dot(rd, hitNormal));

    float3 nro = lAdvance(ro, rd, delta);
    float3 nrd = lReflect(rd, hitNormal);

    Object*   obj = &constants->objects[collisionObject];
    Material* mat = &input.materials[obj->material];

    if (mat->diffuseTexture >= 0) {
      currentColor = prod(currentColor, sampleTexture(&input.textures[mat->diffuseTexture], make_float2(nro.x, nro.z)));
    } else {
      currentColor = prod(currentColor, mat->kd);
    }

    ro = nro;
    rd = nrd;

    float specular = currentColor.x * currentColor.x;
    magic          = lHash(magic);
    rd.x           = rd.x + fresnel * lRandom(lHash(magic)) * 0.02 / specular;
    rd.y           = rd.y + fresnel * lRandom(lHash(magic + 71)) * 0.02 / specular;
    rd.z           = rd.z + fresnel * lRandom(lHash(magic + 45)) * 0.02 / specular;

    rd = lNormalize(rd);
  }

  return prod(currentColor, sampleEnvMap(skyTexture, rd) * 4.0);
}


__global__ void pathTracingKernel(int width, int height, float* fbo_mat, int iterationsPerThread, int maxDepth, SceneInput input, int magic) {

  int    pixelIdx = (blockIdx.y * width + blockIdx.x) * 3;
  float* fbo      = &fbo_mat[blockIdx.z * width * height * 3];

  extern __shared__ float3 sharedResults[];

  float3 partial  = make_float3(0, 0, 0);
  int    tidMagic = ((blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x) * iterationsPerThread + magic;

  for (int i = 0; i < iterationsPerThread; i++) {
    float3 partialResult = pathTracing(width, height, iterationsPerThread, maxDepth, input, blockIdx.x, blockIdx.y, blockIdx.z, tidMagic + i);
    partial.x += partialResult.x;
    partial.y += partialResult.y;
    partial.z += partialResult.z;
  }

  int tid            = threadIdx.x;
  sharedResults[tid] = make_float3(partial.x / iterationsPerThread, partial.y / iterationsPerThread, partial.z / iterationsPerThread);
  __syncthreads();

  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;
    if (index < blockDim.x) {
      sharedResults[index].x += sharedResults[index + s].x;
      sharedResults[index].y += sharedResults[index + s].y;
      sharedResults[index].z += sharedResults[index + s].z;
    }

    __syncthreads();
  }

  if (threadIdx.x == 0) {
    if (input.constants->clear) {
      fbo[pixelIdx]     = (sharedResults[0].x / blockDim.x);
      fbo[pixelIdx + 1] = (sharedResults[0].y / blockDim.x);
      fbo[pixelIdx + 2] = (sharedResults[0].z / blockDim.x);
    } else {
      fbo[pixelIdx] += (sharedResults[0].x / blockDim.x);
      fbo[pixelIdx + 1] += (sharedResults[0].y / blockDim.x);
      fbo[pixelIdx + 2] += (sharedResults[0].z / blockDim.x);
    }
  }
}
static int jobIdCounter = 0;
void       _sceneRun(Scene* scene) {
  dim3 numBlocks           = dim3(scene->desc.frameBufferWidth, scene->desc.frameBufferHeight, scene->desc.framesInFlight);
  int  numThreads          = scene->desc.numThreads;
  int  iterationsPerThread = scene->desc.iterationsPerThread;
  int  jobId               = jobIdCounter;
  dprintf(2, "[CUDA %d ] Running path tracing kernel [%d, %d, %d] with %d threads, iterations per thread: %d\n", jobId, numBlocks.x, numBlocks.y, numBlocks.z, numThreads, iterationsPerThread);
  pathTracingKernel<<<numBlocks, numThreads, sizeof(float3) * numThreads>>>(numBlocks.x, numBlocks.y, (float*)scene->framebuffer.D, iterationsPerThread, scene->desc.rayDepth, sceneInputDevice(scene), lHash(jobIdCounter * 4 + 7));
  dprintf(2, "[CUDA %d ] done\n", jobId);
  jobIdCounter++;
}
extern "C" {
void sceneRun(Scene* scene) {
  _sceneRun(scene);
}


void sceneRunCPU(Scene* scene) {

  int jobId               = jobIdCounter;
  int numThreads          = scene->desc.numThreads;
  int iterationsPerThread = scene->desc.iterationsPerThread;
  dprintf(2, "[CPU %d ] Running path tracing kernel in CPU iterations %d x %d \n", jobId, iterationsPerThread, numThreads);

  int        itCount = 4;
  SceneInput inp     = sceneInputHost(scene);

  for (int i = 0; i < scene->desc.framesInFlight; i++) {
    float* fbo = sceneGetFrame(scene, i);
#pragma omp parallel for
    for (int x = 0; x < scene->desc.frameBufferWidth; x++) {
      for (int y = 0; y < scene->desc.frameBufferHeight; y++) {
        float3 partial = make_float3(0, 0, 0);
        for (int j = 0; j < itCount; j++) {
          int    magicNumber = ((x * scene->desc.frameBufferHeight + y) * scene->desc.frameBufferWidth) * itCount + i;
          float3 result      = pathTracing(scene->desc.frameBufferWidth, scene->desc.frameBufferHeight, numThreads * iterationsPerThread, scene->desc.rayDepth, inp, x, y, i, magicNumber);
          partial.x += result.x;
          partial.y += result.y;
          partial.z += result.z;
        }
        int pixelIdx      = (x * scene->desc.frameBufferWidth + y) * 3;
        fbo[pixelIdx]     = partial.x;
        fbo[pixelIdx + 1] = partial.y;
        fbo[pixelIdx + 2] = partial.z;
      }
    }
  }

  dprintf(2, "[CPU %d ] Done \n", jobId);
  jobIdCounter++;
}
}


// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
  typedef struct
  {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] =
    {
      {0x20, 32},  // Fermi Generation (SM 2.0) GF100 class
      {0x21, 48},  // Fermi Generation (SM 2.1) GF10x class
      {0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
      {0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
      {0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
      {0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
      {0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
      {0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
      {0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
      {0x60, 64},  // Pascal Generation (SM 6.0) GP100 class
      {0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
      {0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one to run properly
  printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}







////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
void sceneScanDevices() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0)
    printf("There is no device supporting CUDA\n");
  int dev;
  for (dev = 0; dev < deviceCount; ++dev) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    if (dev == 0) {
      if (deviceProp.major == 9999 && deviceProp.minor == 9999)
        printf("There is no device supporting CUDA.\n");
      else if (deviceCount == 1)
        printf("There is 1 device supporting CUDA\n");
      else
        printf("There are %d devices supporting CUDA\n", deviceCount);
    }
    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    printf("  Major revision number:                         %d\n",
           deviceProp.major);
    printf("  Minor revision number:                         %d\n",
           deviceProp.minor);
    printf("  Total amount of global memory:                 %zd bytes\n",
           deviceProp.totalGlobalMem);
#if CUDART_VERSION >= 2000
    printf("  Number of multiprocessors:                     %d\n",
           deviceProp.multiProcessorCount);
    printf("  CUDA Cores/MP:                                 %d\n",
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
    printf("  Total CUDA Cores                               %d\n",
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
#endif
    printf("  Total amount of constant memory:               %zd bytes\n",
           deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %zd bytes\n",
           deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",
           deviceProp.warpSize);
    printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
    printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
           deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
           deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("  Maximum memory pitch:                          %zd bytes\n",
           deviceProp.memPitch);
    printf("  Texture alignment:                             %zd bytes\n",
           deviceProp.textureAlignment);
    printf("  Clock rate:                                    %.2f GHz\n",
           deviceProp.clockRate * 1e-6f);
    printf("  Memory Clock rate:                             %.2f GHz\n",
           deviceProp.memoryClockRate * 1e-6f);
    printf("  Memory Bus Width:                              %d bits\n",
           deviceProp.memoryBusWidth);
    printf("  Number of asynchronous engines:                %d\n",
           deviceProp.asyncEngineCount);
    printf("  It can execute multiple kernels concurrently:  %s\n",
           deviceProp.concurrentKernels ? "Yes" : "No");
#if CUDART_VERSION >= 2000
    printf("  Concurrent copy and execution:                 %s\n",
           deviceProp.deviceOverlap ? "Yes" : "No");
#endif
  }
  printf("\nTEST PASSED\n");
}

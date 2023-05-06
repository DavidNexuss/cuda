#include <stdio.h>

extern "C" {
#include "scene.h"
}

#define HEAD __host__ __device__

HEAD float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
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

//Signed distance field functions combined with direction optimisation whenever possible
HEAD int sdfHitSphere(float3 ro, float3 rd, float radius, float* delta, float3* normal);
HEAD int sdfHitPlane(float3 ro, float3 rd, float3 normal, float* delta) {
  if (rd.y > 0.0f) return 0;
  *delta = ro.y / rd.y;
  return 1;
}

HEAD float  fracf(float x) { return x - floorf(x); }
HEAD float2 fracf(float2 uv) { return make_float2(fracf(uv.x), fracf(uv.y)); }
HEAD float  lRandom(int magic) { return fracf(float(magic) * 0.0001); }

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
  float2         uv         = make_float2((2 * y / float(height)) - 1, (2 * (width - x) / float(width)) - 1);
  PushConstants* constants  = input.constants + frame;
  Texture*       skyTexture = &input.textures[constants->uniforms.skyTexture];
  Mesh*          meshes     = input.meshes;

  float3 rd = make_float3(uv.x, uv.y, -1);
  float3 ro = constants->camera.origin;
  //DOF
  magic = lHash(magic);
  rd.x  = rd.x + lRandom(lHash(magic)) * 0.002;
  rd.y  = rd.y + lRandom(lHash(magic + 71)) * 0.002;
  rd.z  = rd.z + lRandom(lHash(magic + 45)) * 0.002;

  rd = lNormalize(rd);

  float3 currentColor = make_float3(1, 1, 1);
  for (int d = 0; d < maxDepth; d++) {
    float3 hitNormal;
    float  delta           = FLT_MAX;
    int    collisionObject = -1;
    for (int i = 0; i < constants->objectCount; i++) {
      Object* obj = &constants->objects[i];
      float   partialDelta;
      float3  partialNormal;
      int     hitStatus;
      switch (meshes[obj->mesh].type) {
        case PLAIN:
          partialNormal = meshes[obj->mesh].tPlain.normal;
          hitStatus     = sdfHitPlane(ro, rd, make_float3(0, 1, 0), &partialDelta);
          break;
        case SPHERE:
          break;
        case MESH:
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
      return prod(currentColor, sampleEnvMap(skyTexture, rd));
    }

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

    magic = lHash(magic);
    rd.x  = rd.x + lRandom(lHash(magic)) * 0.06;
    rd.y  = rd.y + lRandom(lHash(magic + 71)) * 0.04;
    rd.z  = rd.z + lRandom(lHash(magic + 45)) * 0.025;

    rd = lNormalize(rd);
  }
  return currentColor;
}

__global__ void pathTracingKernel(int width, int height, float* fbo_mat, int iterationsPerThread, int maxDepth, SceneInput input) {

  int    pixelIdx = (blockIdx.x * width + blockIdx.y) * 3;
  float* fbo      = &fbo_mat[blockIdx.z * width * height * 3];

  extern __shared__ float3 sharedResults[];
  float3                   partial = make_float3(0, 0, 0);
  int                      tid     = ((blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x) * iterationsPerThread;
  for (int i = 0; i < iterationsPerThread; i++) {
    float3 partialResult = pathTracing(width, height, iterationsPerThread, maxDepth, input, blockIdx.x, blockIdx.y, blockIdx.z, tid + i);
    partial.x += partialResult.x;
    partial.y += partialResult.y;
    partial.z += partialResult.z;
  }

  sharedResults[threadIdx.x] = make_float3(partial.x / iterationsPerThread, partial.y / iterationsPerThread, partial.z / iterationsPerThread);
  __syncthreads();

  //Linear reduction TODO fix this
  if (threadIdx.x == 0) {
    float3 finalResult = make_float3(0, 0, 0);
    for (int i = 0; i < blockDim.x; i++) {
      finalResult.x += sharedResults[i].x;
      finalResult.y += sharedResults[i].y;
      finalResult.z += sharedResults[i].z;
    }
    fbo[pixelIdx]     = finalResult.x / blockDim.x;
    fbo[pixelIdx + 1] = finalResult.y / blockDim.x;
    fbo[pixelIdx + 2] = finalResult.z / blockDim.x;
  }
}
static int jobIdCounter = 0;
void       _sceneRun(Scene* scene) {
  dim3 numBlocks           = dim3(scene->desc.frameBufferWidth, scene->desc.frameBufferHeight, scene->desc.framesInFlight);
  dim3 numThreads          = dim3(scene->desc.numThreads, 1, 1);
  int  iterationsPerThread = scene->desc.iterationsPerThread;
  int  jobId               = jobIdCounter;
  dprintf(2, "[CUDA %d ] Running path tracing kernel [%d, %d, %d] with %d threads, iterations per thread: %d\n", jobId, numBlocks.x, numBlocks.y, numBlocks.z, numThreads.x, iterationsPerThread);
  pathTracingKernel<<<numBlocks, numThreads, sizeof(float3) * numThreads.x>>>(numBlocks.x, numBlocks.y, (float*)scene->framebuffer.D, iterationsPerThread, scene->desc.rayDepth, sceneInputDevice(scene));
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

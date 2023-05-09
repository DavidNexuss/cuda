#include "scripts.hpp"

__device__ int sdBox(float3 ro, float3 rd, float* delta) {

  /*
  float3 q = fabs(ro) - make_float3(1, 1, 1);
  *delta   = length(fmaxf(q, make_float3(0, 0, 0))) + fmin(fmax(q.x, fmax(q.y, q.z)), 0.0f);
  return 1; */

  if (rd.y > 0.0f) return 0;
  *delta = ro.y / rd.y;
  return 1;
}
static __device__ sdfFunction sdfBoxDevice = sdBox;


template <typename T>
T cudaPointer(const T& val) {
  T    f;
  auto ret = cudaMemcpyFromSymbol(&f, val, sizeof(T));
  if (ret != cudaSuccess) {
    dprintf(2, "Error copying symbol to host\n");
    exit(1);
  }

  dprintf(2, "Cuda mempointer %p\n", f);
  return f;
}

extern "C" {
void traceInit(Scene* scene) {
  SceneInput inp = sceneInputHost(scene);

  int meshIdx     = 0;
  int materialIdx = 0;
  int textureIdx  = 0;

  inp.meshes[meshIdx++] = meshCustom(cudaPointer(sdfBoxDevice));

  inp.materials[materialIdx++] = materialCreate(
    make_float3(0.5, 0.7, 0.8),
    make_float3(0.2, 0.4, 0.5),
    make_float3(0.1, 0.1, 0.1),
    0.1,
    1.01);
  inp.materials[0].diffuseTexture = 0;

  inp.textures[textureIdx++] = textureCreate("assets/checker.png");
  inp.textures[textureIdx++] = textureCreate("assets/equi2.png");

  scene->materialCount = materialIdx;
  scene->meshCount     = meshIdx;
  scene->textureCount  = textureIdx;
}

void traceLoop(PushConstants* cn) {
  cn->uniforms.skyColor    = make_float3(0.2, 0.4, 0.9);
  cn->uniforms.groundColor = make_float3(0.2, 0.2, 0.2);
  cn->uniforms.orizonColor = make_float3(0.7, 0.8, 0.9);
  cn->uniforms.skyTexture  = 1;
  cn->camera.znear         = 0.1f;

  cn->camera.up        = make_float3(0, 1, 0);
  cn->camera.origin    = make_float3(0, 0.2, 0);
  cn->camera.direction = make_float3(0, 0, -1);
  cn->camera.crossed   = make_float3(1, 0, 0);

  int objectIdx            = 0;
  cn->objects[objectIdx++] = objectCreate(0, 0, make_float3(0, 0, 1));

  cn->objectCount = objectIdx;
}
}

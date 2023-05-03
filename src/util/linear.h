#pragma once
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

__device__ float3 prod(float3 a, float3 b) { 
  return float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__device__ float3 prod(float3 a, float t) { 
  return prod(a, float3(t,t,t));
}
__device__ float3 sub(float3 a, float3 b) { 
  return float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 sum(float3 a, float3 b) { 
  return float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

//This are cuda versions of basic linear algebra functionality
__device__ float3 lReflect(float3 rd, float3 normal) {}
__device__ float3 lRefract(float3 rd, float3 normal, float ior) {}
__device__ float3 lNormalize(float3 v) {}
__device__ float  lLen2(float3 a) {}
__device__ float  lLen(float3 a) {}

//Returns origin + direction * distance
__device__ float3 lAdvance(float3 origin, float3 direction, float distance) {}

//Retuns a normalized random direction
__device__ float3 lRandomDirection() {}

//Retuns a normalized random direction in a hemisphere from normalvector
__device__ float3 lRrandomDirectionHemisphere(float3 normalvector) {}

//Returns sky color
__device__ dim3 lClearColorBackground(dim3 rd, dim3 ground, dim3 orizon, dim3 sky) {
  float t = rd.y;
  return sub(prod(sky, t), prod(ground, -t));
}


float3 vec3(float r, float g, float b) { return make_float3(r, g, b); }
float3 vec3(float r) { return vec3(r, r, r); }


//Signed distance field functions combined with direction optimisation whenever possible
__device__ bool sdfHitSphere(float3 ro, float3 rd, float radius, float& delta, float3& normal);
__device__ bool sdfHitPlane(float3 ro, float3 rd, float3 normal, float& delta, float& normalDir);

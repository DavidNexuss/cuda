#pragma once
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>


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
__device__ float3 lClearColorBackground(float3 rd, float3 ground, float3 orizon, float3 sky) {}


float3 vec3(float r, float g, float b) { return make_float3(r, g, b); }
float3 vec3(float r) { return vec3(r, r, r); }


//Signed distance field functions combined with direction optimisation whenever possible
__device__ bool sdfHitSphere(float3 ro, float3 rd, float radius, float& delta, float3& normal);
__device__ bool sdfHitPlane(float3 ro, float3 rd, float3 normal, float& delta, float& normalDir);

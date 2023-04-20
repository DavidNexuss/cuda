#pragma once
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>


//This are cuda versions of basic linear algebra functionality
float3 lReflect(float3 rd, float3 normal);
float3 lRefract(float3 rd, float3 normal, float ior);
float3 lNormalize(float3 v);
float  lLen2(float3 a);
float  lLen(float3 a);

//Returns origin + direction * distance
float3 lAdvance(float3 origin, float3 direction, float distance);

//Retuns a normalized random direction
float3 lRandomDirection();

//Retuns a normalized random direction in a hemisphere from normalvector
float3 lRrandomDirectionHemisphere(float3 normalvector);

//Returns distance
float3 lHitSphere(float3 ro, float3 rd, float3 origin, float radius);
float3 lHitPlain(float3 ro, float3 rd, float3 origin, float3 normal);

//Returns normal surface vector
float3 lNormalSphere(float3 ro, float3 rd);
float3 lNormalPlain(float3 ro, float3 rd);

//Returns sky color
float3 clearColorBackground(float3 rd, float3 ground, float3 orizon, float3 sky);

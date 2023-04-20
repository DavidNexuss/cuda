#pragma once

// This are simple cuda linear algebra abstractions
//

struct float3 {
  float x, y, z;
};

float3 reflect(float3 ray, float3 normal);

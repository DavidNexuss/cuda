#pragma once
#include "util/cuda.hpp"
#include "util/linear.hpp"

enum ObjectType {
  PLAIN,
  SPHERE
};

struct Camera {
  float3 origin;
  float3 direction;
};

// Tightly packed ray traced object
struct Object {
  ObjectType type;
  float3     origin;
  union {
    struct {
      float radius;
    } tSphere;

    struct {
      float3 normal;
    } tPlain;
  };
};

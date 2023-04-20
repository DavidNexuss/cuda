#pragma once
#include "util/linear.h"

enum ObjectType {
  PLAIN,
  SPHERE
};

// Camera
typedef struct {
  float3 origin;
  float3 direction;
} Camera;

//Object
typedef struct {
  int    material;
  int    mesh;
  float3 origin;
} Object;

//Material
typedef struct {
  float3 kd;
  float3 ks;
  float3 ka;
  float  fresnel;
  float  ior;
} Material;

// Object
typedef struct {
  enum ObjectType type;
  union {
    struct {
      float radius;
    } tSphere;

    struct {
      float3 normal;
    } tPlain;
  };
} Mesh;

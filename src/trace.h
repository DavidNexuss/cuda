#pragma once
#include "util/linear.h"

// Camera
typedef struct {
  float3 origin;
  float3 direction;
  float3 up;
  float  znear;
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

enum MeshType {
  PLAIN,
  SPHERE
};

// Mesh
typedef struct {
  enum MeshType type;
  union {
    struct {
      float radius;
    } tSphere;

    struct {
      float3 normal;
    } tPlain;
  };
} Mesh;

Mesh meshPlain(float3 normal) {
  Mesh pl;
  pl.type          = PLAIN;
  pl.tPlain.normal = normal;
  return pl;
}

Mesh meshSphere(float radius) {
  Mesh m;
  m.type           = SPHERE;
  m.tSphere.radius = radius;
  return m;
}

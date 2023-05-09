#pragma once
#include "util/linear.h"
// Camera
typedef struct
{
  float3 origin;
  float3 direction;
  float3 up;
  float3 crossed;
  float  znear;
} Camera;

//Object
typedef struct
{
  int    material;
  int    mesh;
  float3 origin;
} Object;

Object objectCreate(int material, int mesh, float3 origin);

//Material
typedef struct
{
  float3 kd;
  float3 ks;
  float3 ka;
  float  fresnel;
  float  ior;
  int    diffuseTexture;
  int    specularTexture;
  int    normalMapTexture;
} Material;

Material materialCreate(float3 kd, float3 ks, float3 ka, float fresnel, float ior);

// Different mesh types
enum MeshType {
  PLAIN,
  SPHERE,
  MESH,
  CUSTOM
};
typedef int (*sdfFunction)(float3 ro, float3 rd, float* delta);
// Mesh varidadic type union over different mesh types
typedef struct
{
  enum MeshType type;
  union {
    struct
    {
      float radius;
    } tSphere;

    struct
    {
      float3 normal;
    } tPlain;

    struct
    {
      int vertexBufferIndex;
      int indexBufferIndex;
      union {
        int vertexCount;
        int indexCount;
      };
    } tMesh;

    struct {
      int         immutableBuffer;
      sdfFunction sdf;
    } tCustom;
  };
} Mesh;

Mesh meshPlain(float3 normal);
Mesh meshSphere(float radius);
Mesh meshCustom(sdfFunction function);

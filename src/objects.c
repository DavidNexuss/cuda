#include "objects.h"
Object   objectCreate(int material, int mesh, float3 origin) {
  Object c;
  c.material = material;
  c.mesh = mesh;
  c.origin = origin;
  return c;
}
Material materialCreate(float3 kd, float3 ks, float3 ka, float fresnel, float ior) {}

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

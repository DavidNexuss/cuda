#include "objects.h"
Object objectCreate(int material, int mesh, float3 origin) {
  Object c;
  c.material = material;
  c.mesh     = mesh;
  c.origin   = origin;
  c.hasTransform = 0;
  return c;
}
Material materialCreate(float3 kd, float3 ks, float3 ka, float fresnel, float ior) {
  Material mat;
  mat.ka      = ka;
  mat.kd      = kd;
  mat.ks      = ks;
  mat.fresnel = fresnel;
  mat.ior     = ior;
  return mat;
}

Mesh meshPlain(float3 normal) {
  Mesh pl          = {};
  pl.type          = PLAIN;
  pl.tPlain.normal = normal;
  return pl;
}

Mesh meshSphere(float radius) {
  Mesh m           = {};
  m.type           = SPHERE;
  m.tSphere.radius = radius;
  return m;
}

Mesh meshCustom(sdfFunction function) {
  Mesh mes;
  mes.type        = CUSTOM;
  mes.tCustom.sdf = function;
  return mes;
}

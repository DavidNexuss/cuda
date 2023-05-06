#include "examples.h"
void defaultScene(Scene* scene) {
  SceneInput inp = sceneInputHost(scene);

  int meshIdx     = 0;
  int materialIdx = 0;
  int textureIdx  = 0;

  inp.meshes[meshIdx++] = meshPlain(make_float3(0, 0, 0));
  inp.meshes[meshIdx++] = meshPlain(make_float3(1, 0, 0));
  inp.meshes[meshIdx++] = meshPlain(make_float3(0, 0, 1));
  inp.meshes[meshIdx++] = meshPlain(make_float3(0, 1, 1));

  Material mat;
  inp.materials[materialIdx++] = materialCreate(
    make_float3(0.5, 0.7, 0.8),
    make_float3(0.2, 0.4, 0.5),
    make_float3(0.1, 0.1, 0.1),
    0.1,
    1.01);

  inp.materials[materialIdx++] = materialCreate(
    make_float3(0.8, 0.7, 0.2),
    make_float3(0.2, 0.2, 0.2),
    make_float3(0.1, 0.1, 0.1),
    0.0,
    1.01);

  inp.materials[materialIdx++] = materialCreate(
    make_float3(10.8, 10.7, 10.2),
    make_float3(0.2, 0.2, 0.2),
    make_float3(0.1, 0.1, 0.1),
    0.0,
    1.01);

  inp.materials[materialIdx++] = materialCreate(
    make_float3(0.01, 0.1, 0.2),
    make_float3(0.8, 1.0, 1.0),
    make_float3(0.1, 0.1, 0.1),
    1.0,
    1.01);

  inp.textures[textureIdx++] = textureCreate("assets/checker.png");
  inp.textures[textureIdx++] = textureCreate("assets/stone.jpg");

  scene->materialCount = materialIdx;
  scene->meshCount     = meshIdx;
  scene->textureCount  = textureIdx;
}

void defaultSceneLoop(PushConstants* cn) {
  cn->uniforms.skyColor    = make_float3(0.2, 0.4, 0.9);
  cn->uniforms.groundColor = make_float3(0.2, 0.2, 0.2);
  cn->uniforms.orizonColor = make_float3(0.7, 0.8, 0.9);

  cn->camera.up        = make_float3(0, 1, 0);
  cn->camera.znear     = 0.1f;
  cn->camera.origin    = make_float3(0, 0, 0);
  cn->camera.direction = make_float3(0, 0, -1);

  int objectIdx            = 0;
  cn->objects[objectIdx++] = objectCreate(0, 1, make_float3(0, 1, 1));
  cn->objects[objectIdx++] = objectCreate(1, 2, make_float3(-1, 1, 1));
  cn->objects[objectIdx++] = objectCreate(2, 3, make_float3(-2, -1, 1));
  cn->objects[objectIdx++] = objectCreate(1, 0, make_float3(2, 1, 1));
  cn->objects[objectIdx++] = objectCreate(3, 0, make_float3(1, -1, 1));

  cn->objectCount = objectIdx;
}
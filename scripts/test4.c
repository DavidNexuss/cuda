#include <scene.h>

const char* testname = "results/test4.png";

void traceInit(Scene* scene) {
  SceneInput inp = sceneInputHost(scene);

  int meshIdx     = 0;
  int materialIdx = 0;
  int textureIdx  = 0;

  inp.meshes[meshIdx++] = meshPlain(make_float3(0, 1, 0));
  inp.meshes[meshIdx++] = meshSphere(1.0);

  inp.materials[materialIdx++] = materialCreate(
    make_float3(0.5, 0.7, 0.8),
    make_float3(0.2, 0.4, 0.5),
    make_float3(0.1, 0.1, 0.1),
    0.1,
    1.01);

  inp.materials[materialIdx++] = materialCreate(
    make_float3(0.5, 0.7, 0.8),
    make_float3(0.2, 0.4, 0.5),
    make_float3(0.1, 0.1, 0.1),
    0.1,
    1.01);

  inp.materials[materialIdx++] = materialCreate(
    make_float3(0.5, 0.7, 0.8),
    make_float3(0.2, 0.4, 0.5),
    make_float3(0.1, 0.1, 0.1),
    0.1,
    1.01);

  inp.materials[0].diffuseTexture = 2;
  inp.materials[1].diffuseTexture = 0;
  inp.materials[2].diffuseTexture = 3;


  inp.textures[textureIdx++] = textureCreate("assets/checker.png");
  inp.textures[textureIdx++] = textureCreate("assets/envMap.jpg");
  inp.textures[textureIdx++] = textureCreate("assets/stone.jpg");
  inp.textures[textureIdx++] = textureCreate("assets/quartz.jpg");

  scene->materialCount = materialIdx;
  scene->meshCount     = meshIdx;
  scene->textureCount  = textureIdx;
}

void traceLoop(PushConstants* cn) {
  cn->uniforms.skyColor    = make_float3(0.2, 0.4, 0.9);
  cn->uniforms.groundColor = make_float3(0.2, 0.2, 0.2);
  cn->uniforms.orizonColor = make_float3(0.7, 0.8, 0.9);
  cn->uniforms.skyTexture  = 1;
  cn->camera.znear         = 0.1f;

  cn->camera.up        = make_float3(0, 1, 0);
  cn->camera.origin    = make_float3(0, 0.2, 0);
  cn->camera.direction = make_float3(0, 0, -1);
  cn->camera.crossed   = make_float3(1, 0, 0);

  int objectIdx            = 0;
  cn->objects[objectIdx++] = objectCreate(0, 0, make_float3(0, 0, 1));
  cn->objects[objectIdx++] = objectCreate(1, 1, make_float3(0, 0, -2));
  cn->objects[objectIdx++] = objectCreate(2, 1, make_float3(-1, 0, -3));
  cn->objects[objectIdx++] = objectCreate(2, 1, make_float3(1, 0, -3));

  cn->objectCount = objectIdx;
}

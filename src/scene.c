#include "scene.h"
#include "texture.h"
#include "util/buffer.h"
#include <stb/stb_image_write.h>
Scene sceneCreate(SceneDesc desc) {
  Scene scene;
  scene.desc        = desc;
  scene.meshes      = bufferCreate(sizeof(Mesh) * desc.maxMeshes);
  scene.materials   = bufferCreate(sizeof(Material) * desc.maxMaterials);
  scene.framebuffer = bufferCreate(3 * sizeof(float) * desc.frameBufferWidth * desc.frameBufferHeight * desc.framesInFlight);
  scene.constants   = bufferCreate(sizeof(PushConstants) * desc.framesInFlight);

  scene.texturesTable     = bufferCreate(sizeof(Texture) * desc.maxTextures);
  scene.vertexBufferTable = bufferCreate(sizeof(BufferObject) * desc.maxVertexBuffer);
  scene.indexBufferTable  = bufferCreate(sizeof(BufferObject) * desc.maxIndexBuffer);

  scene.materialCount     = -1;
  scene.textureCount      = -1;
  scene.meshCount         = -1;
  scene.indexBufferCount  = -1;
  scene.vertexBufferCount = -1;

  scene.objects = (Buffer*)malloc(sizeof(Buffer) * desc.framesInFlight);

  PushConstants* cn = sceneInputHost(&scene).constants;
  for (int i = 0; i < desc.framesInFlight; i++) {
    scene.objects[i] = bufferCreate(sizeof(Object) * desc.maxObjects);
    cn[i].objects    = (Object*)scene.objects[i].H;
  }

  // Late
  scene.vertexBuffersData = (void**)malloc(sizeof(void*) * desc.maxVertexBuffer);
  scene.indexBuffersData  = (void**)malloc(sizeof(void*) * desc.maxIndexBuffer);
  scene.texturesData      = (void**)malloc(sizeof(void*) * desc.maxTextures);

  for (int i = 0; i < desc.maxTextures; i++) scene.texturesData[i] = 0;

  return scene;
}

/* returns host bag of pointers */
SceneInput sceneInputHost(Scene* scene) {
  SceneInput inp;
  inp.materials     = (Material*)scene->materials.H;
  inp.meshes        = (Mesh*)scene->meshes.H;
  inp.textures      = (Texture*)scene->texturesTable.H;
  inp.constants     = (PushConstants*)scene->constants.H;
  inp.indexBuffers  = (BufferObject*)scene->indexBufferTable.H;
  inp.vertexBuffers = (BufferObject*)scene->vertexBufferTable.H;
  return inp;
}

SceneInput sceneInputDevice(Scene* scene) {
  SceneInput inp;
  inp.materials     = (Material*)scene->materials.D;
  inp.meshes        = (Mesh*)scene->meshes.D;
  inp.textures      = (Texture*)scene->texturesTable.D;
  inp.constants     = (PushConstants*)scene->constants.D;
  inp.indexBuffers  = (BufferObject*)scene->indexBufferTable.D;
  inp.vertexBuffers = (BufferObject*)scene->vertexBufferTable.D;
  return inp;
}

/* destroys scene and releases memory */
void sceneDestroy(Scene* scene) {
  bufferDestroy(&scene->meshes);
  bufferDestroy(&scene->materials);
  bufferDestroy(&scene->framebuffer);

  {
    for (int i = 0; i < scene->textureCount; i++) {
      Texture* textureTable = scene->texturesTable.H;
      textureDestroy(&textureTable[i]);
      bufferDestroyImmutable(scene->texturesData[i]);
    }
    bufferDestroy(&scene->texturesTable);
    free(scene->texturesData);
  }

  for (int i = 0; i < scene->desc.framesInFlight; i++) {
    bufferDestroy(&scene->objects[i]);
  }

  bufferDestroy(&scene->constants);
  free(scene->objects);

  //Late

  for (int i = 0; i < scene->vertexBufferCount; i++) {
    BufferObject* objects = scene->vertexBufferTable.H;
    free(objects[i].data);
    bufferDestroyImmutable(&scene->vertexBuffersData[i]);
  }

  for (int i = 0; i < scene->indexBufferCount; i++) {
    BufferObject* objects = scene->indexBufferTable.H;
    free(objects[i].data);
    bufferDestroyImmutable(&scene->indexBuffersData[i]);
  }

  bufferDestroy(&scene->vertexBufferTable);
  bufferDestroy(&scene->indexBufferTable);
  free(scene->vertexBuffersData);
  free(scene->indexBuffersData);
}

/* uploads scene */
void sceneUpload(Scene* scene) {
  bufferUploadAmount(&scene->materials, scene->materialCount * sizeof(Material));
  bufferUploadAmount(&scene->meshes, scene->meshCount * sizeof(Mesh));

  void**     tmp = malloc(sizeof(void*) * scene->textureCount);
  SceneInput inp = sceneInputHost(scene);
  for (int i = 0; i < scene->textureCount; i++) {
    Texture t              = inp.textures[i];
    scene->texturesData[i] = bufferCreateImmutable(t.data, t.channels * t.height * t.width);
    tmp[i]                 = inp.textures[i].data;
    inp.textures[i].data   = scene->texturesData[i];
  }

  bufferUploadAmount(&scene->texturesTable, scene->textureCount * sizeof(Texture));

  for (int i = 0; i < scene->textureCount; i++) {
    inp.textures[i].data = tmp[i];
  }

  free(tmp);
}

void sceneUploadObjects(Scene* scene) {
  //Upload required buffer objects for each frame
  PushConstants* c = sceneInputHost(scene).constants;
  {
    for (int i = 0; i < scene->desc.framesInFlight; i++) {
      bufferUploadAmount(&scene->objects[i], c[i].objectCount * sizeof(Object));
    }
  }

  //Upload push constants for each frame
  {
    for (int i = 0; i < scene->desc.framesInFlight; i++) {
      c[i].objects = (Object*)scene->objects[i].D;
    }
    bufferUpload(&scene->constants);

    for (int i = 0; i < scene->desc.framesInFlight; i++) {
      c[i].objects = (Object*)scene->objects[i].H;
    }
  }
}

PushConstants* scenePushConstantsHost(Scene* scene) {
  return (PushConstants*)scene->constants.H;
}
PushConstants* scenePushConstantsDevice(Scene* scene) {
  return (PushConstants*)scene->constants.D;
}

void sceneDownload(Scene* scene) {
  bufferDownload(&scene->framebuffer);
}

float* sceneGetFrame(Scene* scene, int index) {
  float* fbo = (float*)scene->framebuffer.H;
  return &fbo[index * 3 * scene->desc.frameBufferWidth * scene->desc.frameBufferHeight];
}

#include <stdio.h>
void sceneWriteFrame(Scene* scene, const char* path, int index) {
  dprintf(2, "[IO] Writing frame [%d] for scene to %s\n", index, path);
  if (scene->desc.fWriteClamped) {

    int            count = scene->desc.frameBufferWidth * scene->desc.frameBufferHeight * 3;
    float*         fbo   = sceneGetFrame(scene, index);
    unsigned char* png   = (unsigned char*)malloc(count);

    float maxValue = 0.0f;
    float minValue = 10000000.0f;
    for (int i = 0; i < count; i++) {
      if (fbo[i] > maxValue) maxValue = fbo[i];
      if (fbo[i] < minValue) minValue = fbo[i];
    }

    dprintf(2, "Max and min values %f %f\n", maxValue, minValue);
    if ((maxValue - minValue) < 0.01) { maxValue += 0.2; }
    for (int i = 0; i < count; i++) {
      png[i] = ((fbo[i] - minValue) / (maxValue - minValue)) * 0xff;
    }

    stbi_write_png(path, scene->desc.frameBufferWidth, scene->desc.frameBufferHeight, 3, png, scene->desc.frameBufferWidth * 3);
    free(png);
  } else {
    stbi_write_hdr(path, scene->desc.frameBufferWidth, scene->desc.frameBufferHeight, 3, sceneGetFrame(scene, index));
  }
}

void sceneRunSuite(SceneDesc sceneDesc, const char* path, void(initScene)(Scene*), void(initSceneFrame)(PushConstants* cn), int cpu) {

  Scene scene = sceneCreate(sceneDesc);

  //Inits scene materials and meshes
  {
    initScene(&scene);
    if (!cpu) sceneUpload(&scene);
  }

  //Inits scene objects
  {
    SceneInput sc = sceneInputHost(&scene);
    float      t  = 0;

    for (int i = 0; i < sceneDesc.framesInFlight; i++) {
      PushConstants* constants = &sc.constants[i];
      constants->frameTime     = t;
      initSceneFrame(constants);
    }

    if (!cpu) sceneUploadObjects(&scene);
  }

  if (!cpu) sceneRun(&scene);
  if (!cpu) sceneDownload(&scene);

  if (cpu) sceneRunCPU(&scene);
  for (int i = 0; i < sceneDesc.framesInFlight; i++) {
    sceneWriteFrame(&scene, path, i);
  }
  sceneDestroy(&scene);
}

#include <math.h>

inline static __host__ __device__ float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
void sceneRunSuiteMovie(SceneDesc sceneDesc, const char* path, void(initScene)(Scene*), void(initSceneFrame)(PushConstants* cn)) {

  int simultaneous         = 32;
  int maxIterations        = (32 * 3) / simultaneous;
  sceneDesc.framesInFlight = simultaneous;
  Scene scene              = sceneCreate(sceneDesc);

  //Inits scene materials and meshes
  {
    initScene(&scene);
    sceneUpload(&scene);
  }

  float t = 0;
  char  buffer[256];
  //Inits scene objects
  int f = 0;
  for (int d = 0; d < maxIterations + 1; d++) {
    dprintf(2, "[%d / %d] Running iteration\n", d, maxIterations);
    SceneInput sc = sceneInputHost(&scene);

    for (int i = 0; i < sceneDesc.framesInFlight; i++) {
      PushConstants* constants = &sc.constants[i];
      initSceneFrame(constants);
      constants->frameTime        = t;
      constants->camera.direction = make_float3(sin(t), 0, cos(t));
      constants->camera.crossed   = cross(constants->camera.up, constants->camera.direction);
      constants->clear            = 1;
      t += 0.01f;
    }

    sceneUploadObjects(&scene);
    sceneRun(&scene);

    if (d != 0) {
#pragma omp parallel for
      for (int i = 0; i < sceneDesc.framesInFlight; i++) {
        sprintf(buffer, "%s_%03d.png", path, f + i);
        sceneWriteFrame(&scene, buffer, i);
      }

      f += sceneDesc.framesInFlight;
    }

    sceneDownload(&scene);
  }

  sceneDestroy(&scene);
}

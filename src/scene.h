#pragma once

#include "util/buffer.h"
#include "objects.h"
#include "texture.h"

typedef struct {
  dim3 skyColor;
  dim3 orizonColor;
  dim3 groundColor;
} RenderingUniforms;

/*
 This struct defines all configurable parameters for class scene
*/
typedef struct {
  int   maxObjects;
  int   maxMaterials;
  int   maxMeshes;
  int   maxTextures;
  int   frameBufferWidth;
  int   frameBufferHeight;
  int   numThreads;
  int   iterationsPerThread;
  int   rayDepth;
  int   framesInFlight;
  float frameDelta;
} SceneDesc;

typedef struct {
  RenderingUniforms uniforms;
  Camera            camera;
  float             frameTime;
  int               objectCount;
  Object*           objects;
} PushConstants;

/* Execution model
 * Grid = dim3(width, height, framesInFlight)
 * Block = dim3(numThreads, 1, 1)
 *
 * 1 framesInFlight
 * N framesInFlight with reduction kernel
 * N framesInFlight with different frames and/or reduction
 * N framesInFlight with different frames multiple GPUs (scenes) and/or reduction
 */

typedef struct _Scene {
  //Input buffer objects
  Buffer meshes;
  Buffer materials;
  Buffer textures;

  //Output buffer objects
  Buffer framebuffer;

  //Uniforms
  Buffer* objects; //List of buffers, one per each frame in flight
  Buffer  constants;

  int materialCount;
  int meshCount;
  int textureCount;

  //Scene configuration
  SceneDesc desc;

} Scene;

/* bag of pointers */
typedef struct {
  Material*      materials;
  Mesh*          meshes;
  Texture*       textures;
  PushConstants* constants;
} SceneInput;


Scene      sceneCreate(SceneDesc desc);
SceneInput sceneInputHost(Scene* scene);
SceneInput sceneInputDevice(Scene* scene);
void       sceneDestroy(Scene* scene);
void       sceneUpload(Scene* scene);
void       sceneUploadObjects(Scene* scene);
void       sceneDownload(Scene* scene);
float*     sceneGetFrame(Scene* scene, int index);
void       sceneWriteFrame(Scene* scene, const char* path, int index);

void sceneRun(Scene* scene);

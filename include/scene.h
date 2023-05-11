#pragma once

#include "util/buffer.h"
#include "bufferObject.h"
#include "objects.h"
#include "texture.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
  float3 skyColor;
  float3 orizonColor;
  float3 groundColor;
  int    skyTexture;
} RenderingUniforms;

/*
 This struct defines all configurable parameters for class scene
*/
typedef struct
{

  //Satic sized allocated buffers, ugly but comfortable
  int maxObjects;
  int maxMaterials;
  int maxMeshes;
  int maxTextures;
  int maxVertexBuffer;
  int maxIndexBuffer;

  //Framebuffer definition
  int frameBufferWidth;
  int frameBufferHeight;
  int framesInFlight;

  //Cuda algorithm parameters
  int   numThreads;
  int   iterationsPerThread;
  int   rayDepth;
  float frameDelta;

  int fWriteClamped;
} SceneDesc;

typedef struct
{
  RenderingUniforms uniforms;
  Camera            camera;
  float             frameTime;
  int               objectCount;
  Object*           objects;
  int               clear;
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

  //Output buffer objects
  Buffer framebuffer;

  //Uniforms
  Buffer* objects;
  Buffer  constants;

  int materialCount;
  int meshCount;

  //Scene configuration
  SceneDesc desc;

  //Late
  void** texturesData;
  void** vertexBuffersData;
  void** indexBuffersData;

  Buffer texturesTable;
  Buffer vertexBufferTable;
  Buffer indexBufferTable;

  int vertexBufferCount;
  int indexBufferCount;
  int textureCount;

  int _backendNeedsUpdate;
  int _backendNeedsObjectUpdate;

} Scene;

/* bag of pointers */
typedef struct
{
  Material*      materials;
  Mesh*          meshes;
  Texture*       textures;
  PushConstants* constants;
  BufferObject*  vertexBuffers;
  BufferObject*  indexBuffers;
} SceneInput;


void           hintUseWindow(int fUseWindow);
Scene          sceneCreate(SceneDesc desc);
SceneInput     sceneInputHost(Scene* scene);
SceneInput     sceneInputDevice(Scene* scene);
void           sceneDestroy(Scene* scene);
void           sceneUpload(Scene* scene);
void           sceneUploadObjects(Scene* scene);
void           sceneDownload(Scene* scene);
float*         sceneGetFrame(Scene* scene, int index);
void           sceneWriteFrame(Scene* scene, const char* path, int index);
unsigned char* scenePng(Scene* scene, int index);
int            sceneLoadOBJ(const char* path, Scene* scene);


void sceneRun(Scene* scene);
void sceneRunCPU(Scene* scene);
void sceneRunCPUMultiThreaded(Scene* scene);

void sceneRunSuite(SceneDesc sceneDesc, const char* path, void(initScene)(Scene*), void(initSceneFrame)(PushConstants* cn), int cpu);

void sceneRunSuiteMovie(SceneDesc sceneDesc, const char* path, void(initScene)(Scene*), void(initSceneFrame)(PushConstants* cn), void(callback)(Scene*, int, const char* path));
void sceneRunSuiteMovieFrames(SceneDesc sceneDesc, const char* path, void(initScene)(Scene*), void(initSceneFrame)(PushConstants* cn));
void sceneRunSuiteMovieEncode(SceneDesc sceneDesc, const char* path, void(initScene)(Scene*), void(initSceneFrame)(PushConstants* cn));
#ifdef __cplusplus
}
#endif

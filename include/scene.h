#pragma once

#include "util/buffer.h"
#include "bufferObject.h"
#include "objects.h"
#include "texture.h"

/* Rendering uniforms */
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

  /* Flag to indicate wether or not to output PNG or HDR file */
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


/* Memory model
 * For each frame in flight we have one struct of PushConstants. For each PushConstants bag we have a buffer
 * of objects that maps to a list of Objects of our scene.
 *
 * For each scene we have a buffer for meshes, materials, texturens and framebuffers (one per each frame in flight)
 * We also have a list of buffers for vertexBuffers, indexBuffers and texturesData
 *
 * vertexBuffers and indexBuffers remain unimplemented in the tracing algorithm but the model implementation is kept
 *
 * Textures work in the following way, each texture has an individual buffer allocated in the GPU for its texel data, alongside
 * each texture has a record in a global buffer descripting its meta data (width height and channels) so the GPU can sample the data.
 *
 * Materials and Objects keep references to Textures and Materials using locations in its respective table, which is defined as a pointer
 *
 * Seperation between Objects and Materials and Textures through PushConstants interface is done so multiple frames can have its own version of the Object
 * List allowing for generating animations or multiple version of the same frame (hence using the code to improve quality or to improve parallelism)
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

} Scene;

/* bag of pointers */
typedef struct
{
  Material*      materials;     /* Table of materials*/
  Mesh*          meshes;        /* Table of meshes */
  Texture*       textures;      /* Table of textures */
  PushConstants* constants;     /* Table of PushConstants (one per frame) */
  BufferObject*  vertexBuffers; /* Table of vertexBuffers (not used in tracing algorithm) WIP*/
  BufferObject*  indexBuffers;  /* Table of indexBuffers (not used in tracing algorithm) WIP*/
} SceneInput;


/* Scene related functions */
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

/* Scene runners */

void sceneRun(Scene* scene);                 /* Default cuda runner */
void sceneRunCPU(Scene* scene);              /* CPU runner */
void sceneRunCPUMultiThreaded(Scene* scene); /* CPU runner with openmp support */


/* Scene movie generation and batch runners */
void sceneRunSuite(SceneDesc sceneDesc, const char* path, void(initScene)(Scene*), void(initSceneFrame)(PushConstants* cn), int cpu);
void sceneRunSuiteMovie(SceneDesc sceneDesc, const char* path, void(initScene)(Scene*), void(initSceneFrame)(PushConstants* cn), void(callback)(Scene*, int, const char* path));
void sceneRunSuiteMovieFrames(SceneDesc sceneDesc, const char* path, void(initScene)(Scene*), void(initSceneFrame)(PushConstants* cn));
void sceneRunSuiteMovieEncode(SceneDesc sceneDesc, const char* path, void(initScene)(Scene*), void(initSceneFrame)(PushConstants* cn));


void sceneScanDevices();

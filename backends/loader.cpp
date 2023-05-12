#include "texture.h"
#include <cstdio>
#include <scene.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "backend.h"

#include <iostream>
#include <unordered_map>
struct LoaderCache {
  std::unordered_map<int, int>         meshCache;
  std::unordered_map<std::string, int> textureCache;
  std::unordered_map<int, int>         materialCache;

  void clear() {
    meshCache.clear();
    textureCache.clear();
    materialCache.clear();
  }
};

LoaderCache cache;

static int processTexture(const std::string& path, Scene* tracerScene) {
  auto it = cache.textureCache.find(path);
  if (it != cache.textureCache.end()) return it->second;

  if (tracerScene->textureCount <= 0) tracerScene->textureCount = 0;
  int      texture  = tracerScene->textureCount;
  Texture* textures = (Texture*)tracerScene->texturesTable.H;

  textures[texture] = textureCreate(path.c_str());

  tracerScene->textureCount++;

  dprintf(2, "[LOADER] Texture created %d\n", texture);
  return cache.textureCache[path] = texture;
}
static int processMaterial(int materialIdx, const aiScene* scene, Scene* tracerScene) {
  auto it = cache.materialCache.find(materialIdx);
  if (it != cache.materialCache.end()) return it->second;

  aiMaterial* mat = scene->mMaterials[materialIdx];

  if (tracerScene->materialCount <= 0) tracerScene->materialCount = 0;
  int       traceMaterial    = tracerScene->materialCount;
  Material* traceMaterialPtr = ((Material*)tracerScene->materials.H) + traceMaterial;

  mat->Get(AI_MATKEY_COLOR_DIFFUSE, traceMaterialPtr->kd);
  mat->Get(AI_MATKEY_COLOR_AMBIENT, traceMaterialPtr->ka);
  mat->Get(AI_MATKEY_COLOR_SPECULAR, traceMaterialPtr->ks);

  static aiTextureType types[] = {aiTextureType_DIFFUSE, aiTextureType_SPECULAR};
  //TODO: Magic number
  for (int type = 0; type < 2; type++) {
    for (int i = 0; i < mat->GetTextureCount(types[type]); i++) {
      aiString path;
      mat->GetTexture(types[type], i, &path);
      switch (type) {
        case 0: traceMaterialPtr->diffuseTexture = processTexture(path.C_Str(), tracerScene); break;
        case 1: traceMaterialPtr->specularTexture = processTexture(path.C_Str(), tracerScene); break;
      }
    }
  }

  tracerScene->materialCount++;

  dprintf(2, "[LOADER] Material created %d\n", traceMaterial);
  return cache.materialCache[materialIdx] = traceMaterial;
}
static int processMesh(int meshIdx, const aiScene* scene, Scene* tracerScene) {

  auto it = cache.meshCache.find(meshIdx);
  if (it != cache.meshCache.end()) return it->second;

  aiMesh* mesh = scene->mMeshes[meshIdx];
  if (tracerScene->vertexBufferCount <= 0) tracerScene->vertexBufferCount = 0;
  if (tracerScene->indexBufferCount <= 0) tracerScene->indexBufferCount = 0;

  int vbo = tracerScene->vertexBufferCount;
  int ebo = tracerScene->indexBufferCount;

  int vbo_count = mesh->mNumVertices;

  float* vbo_data = (float*)malloc(VERTEX_SIZE * sizeof(float) * vbo_count);


  int t = 0;
  for (int i = 0; i < vbo_count; i++) {
    vbo_data[t++] = mesh->mVertices[i].x;
    vbo_data[t++] = mesh->mVertices[i].y;
    vbo_data[t++] = mesh->mVertices[i].z;

    vbo_data[t++] = mesh->mNormals[i].x;
    vbo_data[t++] = mesh->mNormals[i].y;
    vbo_data[t++] = mesh->mNormals[i].z;

    if (mesh->mTextureCoords[0] != 0) {
      vbo_data[t++] = mesh->mTextureCoords[0][i].x;
      vbo_data[t++] = mesh->mTextureCoords[0][i].y;
    } else {
      vbo_data[t++] = 0;
      vbo_data[t++] = 0;
    }
  }
  if (t != (vbo_count * VERTEX_SIZE)) {
    dprintf(2, "VBO Buffer overflow\n");
    exit(1);
  }

  int ebo_count = 0;
  for (int i = 0; i < mesh->mNumFaces; i++) { ebo_count += mesh->mFaces[i].mNumIndices; }

  unsigned int* ebo_data = (unsigned int*)malloc(sizeof(unsigned int) * ebo_count);

  t = 0;
  for (int i = 0; i < mesh->mNumFaces; i++) {
    for (int j = 0; j < mesh->mFaces[i].mNumIndices; j++) {
      ebo_data[t++] = mesh->mFaces[i].mIndices[j];
    }
  }

  if (t != (ebo_count)) {
    dprintf(2, "EBO Buffer overflow\n");
    exit(1);
  }

  if (tracerScene->meshCount <= 0) tracerScene->meshCount = 0;
  int meshTracer = tracerScene->meshCount;

  SceneInput in = sceneInputHost(tracerScene);

  in.meshes[meshTracer].type                    = MESH;
  in.meshes[meshTracer].tMesh.indexCount        = ebo_count;
  in.meshes[meshTracer].tMesh.vertexCount       = vbo_count;
  in.meshes[meshTracer].tMesh.vertexBufferIndex = vbo;
  in.meshes[meshTracer].tMesh.indexBufferIndex  = ebo;

  tracerScene->vertexBuffersData[vbo] = vbo_data;
  tracerScene->indexBuffersData[ebo]  = ebo_data;

  tracerScene->meshCount++;
  tracerScene->indexBufferCount++;
  tracerScene->vertexBufferCount++;

  dprintf(2, "[LOADER] Mesh created %d\n", meshTracer);
  return cache.meshCache[meshIdx] = meshTracer;
}

void toArray(Mat4 array, aiMatrix4x4& n) {
  array[0] = make_float4(n.a1, n.a2, n.a3, n.a4);
  array[1] = make_float4(n.b1, n.b2, n.b3, n.b4);
  array[2] = make_float4(n.c1, n.c2, n.c3, n.c4);
  array[3] = make_float4(n.d1, n.d2, n.d3, n.d4);
}
static void processNode(aiNode* node, const aiScene* scene, Scene* tracerScene, aiMatrix4x4 parentTransform) {

  aiMatrix4x4 nodeTransform = parentTransform * node->mTransformation;

  SceneInput in = sceneInputHost(tracerScene);
  for (unsigned int i = 0; i < node->mNumMeshes; i++) {
    int meshTracer    = processMesh(node->mMeshes[i], scene, tracerScene);
    int materialTrace = processMaterial(scene->mMeshes[node->mMeshes[i]]->mMaterialIndex, scene, tracerScene);
    for (int i = 0; i < tracerScene->desc.framesInFlight; i++) {
      PushConstants* cn = in.constants + i;
      if (cn->objectCount <= 0) cn->objectCount = 0;
      int               obj = cn->objectCount;
      aiVector3t<float> pos, scale, rot;
      node->mTransformation.Decompose(scale, rot, pos);
      cn->objects[obj].mesh     = meshTracer;
      cn->objects[obj].material = materialTrace;
      cn->objects[obj].origin   = make_float3(pos.x, pos.y, pos.z);
      cn->objects->hasTransform = 1;
      toArray(cn->objects[obj].transformMatrix, nodeTransform);
      cn->objectCount++;
      dprintf(2, "[LOADER] Object created %d\n", obj);
    }
  }
  // then do the same for each of its children
  for (unsigned int i = 0; i < node->mNumChildren; i++) {
    processNode(node->mChildren[i], scene, tracerScene, nodeTransform);
  }
}

static int _sceneLoadOBJ(const char* path, Scene* scene) {
  cache.clear();
  Assimp::Importer importer;
  const aiScene*   scn = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);
  if (!scn || scn->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scn->mRootNode) {
    dprintf(2, "[LOADER] Error processing mesh\n");
    return 1;
  }
  aiMatrix4x4 id;
  id.a1 = 1;
  id.b2 = 1;
  id.c3 = 1;
  id.d4 = 1;
  processNode(scn->mRootNode, scn, scene, id);
  dprintf(2, "[LOADER] Mesh processing successful\n");
  return 0;
}

extern "C" {
int sceneLoadOBJ(const char* path, Scene* scene) { return _sceneLoadOBJ(path, scene); }
}

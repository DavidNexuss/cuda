#include <scene.h>


typedef struct { 
  float* vbo;
  unsigned int* ebo;
  int numVertices;
  int numIndices;
} MeshDesc;

int sceneAddMesh(Scene* scene, MeshDesc desc) {
  
  if(scene->meshCount <= 0) scene->meshCount = 0;
  if(scene->vertexBufferCount<= 0) scene->vertexBufferCount= 0;
  if(scene->indexBufferCount <= 0) scene->indexBufferCount= 0;

  int mesh = scene->meshCount;
  scene->vertexBuffersData[scene->vertexBufferCount] = desc.vbo;
  scene->indexBuffersData[scene->indexBufferCount] = desc.ebo;
  Mesh* meshList = (Mesh*)scene->meshes.H;
  meshList[mesh].type = MESH;
  meshList[mesh].tMesh.vertexBufferIndex = scene->vertexBufferCount;
  meshList[mesh].tMesh.indexBufferIndex = scene->indexBufferCount;
  meshList[mesh].tMesh.vertexCount = desc.numVertices;
  meshList[mesh].tMesh.indexCount = desc.numIndices;

  scene->vertexBufferCount++;
  scene->indexBufferCount++;
  scene->meshCount++;
  return mesh;
}

#include <vector>
#include <cmath>
using namespace std;
struct Vertex { 
  float3 pos;
  float3 normal;
  float2 uv;
};

template <typename T>
void vectorAdd(vector<T>& dst, const vector<T>& sink, int base) { for(auto& d : sink) dst.push_back(d + base); }

inline __host__ float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

static float  lLen2(float3 a) {
  return a.x * a.x + a.y * a.y + a.z * a.z;
}
static float lLen(float3 a) {
  return sqrt(lLen2(a));
}

static float3 lNormalize(float3 v) {
  float len = lLen(v);
  return make_float3(v.x / len, v.y / len, v.z / len);
}
void addPlain(std::vector<Vertex>& vbo, std::vector<unsigned int>& ebo, float2 start, float2 end, float H) {

  int b = vbo.size();
  auto d= make_float2(end.x - start.x, end.y - start.y);
  float Di = std::sqrt(d.x * d.x + d.y * d.y);

  float3 A = make_float3(start.x, 0, start.y);
  float3 B = make_float3(start.x, H, start.y);
  float3 C = make_float3(end.x, H, end.y);
  float3 D = make_float3(end.x, 0, end.y);
  
  float3 N = cross(make_float3(B.x - A.x, B.y - A.y, B.z - A.z), make_float3(C.x - A.x, C.y - A.y, C.z - A.z));
  N = lNormalize(N);

  vbo.push_back({A,N, make_float2(0,0)});
  vbo.push_back({B,N, make_float2(0,H)});
  vbo.push_back({C,N, make_float2(Di,H)});
  vbo.push_back({D,N, make_float2(Di,0)});

  vectorAdd(ebo, {0,2,1,3,2,0}, b);
}

void addCube(std::vector<Vertex>& vbo, std::vector<unsigned int>& ebo, float H, float2 offset, float S) { 

  addPlain(vbo, ebo, make_float2(0 + offset.x,0 + offset.y), make_float2(0 + offset.x,S + offset.y), H);
  addPlain(vbo, ebo, make_float2(0 + offset.x,S + offset.y), make_float2(S + offset.x,S + offset.y), H);
  addPlain(vbo, ebo, make_float2(S + offset.x,S + offset.y), make_float2(S + offset.x,0 + offset.y), H);
  addPlain(vbo, ebo, make_float2(S + offset.x,0 + offset.y), make_float2(0 + offset.x,0 + offset.y), H);
}

MeshDesc getMesh(int count) { 
  std::vector<Vertex>* vbo = new std::vector<Vertex>();
  std::vector<unsigned int>* ebo = new std::vector<unsigned int>();


  for(int i = 0; i < count; i++) { 
    for(int j = 0; j < count; j++) { 
      float H = (i + j) % 5 + 10;
      addCube(*vbo, *ebo, H, make_float2((i - count/2) * 10,(j - count/2) * 10), 5);
      // addCube(*vbo, *ebo, 1, make_float2(i * 10 - 0.5,j * 10 - 0.5), 6);
    }
  }

  MeshDesc desc;
  desc.numVertices = vbo->size();
  desc.numIndices = ebo->size();
  desc.vbo = (float*)vbo->data();
  desc.ebo = ebo->data();
  return desc;
}
extern "C"
void traceInit(Scene* scene) {
  SceneInput inp = sceneInputHost(scene);

  int meshIdx     = 0;
  int materialIdx = 0;
  int textureIdx  = 0;
  int vboIdx= 0;
  int eboIdx= 0;

  inp.meshes[meshIdx++] = meshPlain(make_float3(0, 1, 0));
  scene->meshCount = meshIdx;
  sceneAddMesh(scene, getMesh(100));
  meshIdx = scene->meshCount;

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

  inp.materials[0].diffuseTexture = 0;
  inp.materials[1].diffuseTexture = 2;

  inp.textures[textureIdx++] = textureCreate("assets/stone.jpg");
  inp.textures[textureIdx++] = textureCreate("assets/envMap.jpg");
  inp.textures[textureIdx++] = textureCreate("assets/checker.png");


  scene->materialCount = materialIdx;
  scene->meshCount     = meshIdx;
  scene->textureCount  = textureIdx;
  scene->vertexBufferCount = vboIdx;
  scene->indexBufferCount = eboIdx;
}
extern "C"
void traceLoop(PushConstants* cn) {
  cn->uniforms.skyColor    = make_float3(0.2, 0.4, 0.9);
  cn->uniforms.groundColor = make_float3(0.2, 0.2, 0.2);
  cn->uniforms.orizonColor = make_float3(0.7, 0.8, 0.9);
  cn->uniforms.skyTexture  = 1;
  cn->camera.znear         = 0.1f;

  cn->camera.up        = make_float3(0, 1, 0);
  cn->camera.origin    = make_float3(0, 1, 0);
  cn->camera.direction = make_float3(0, 0, -1);
  cn->camera.crossed   = make_float3(1, 0, 0);

  int objectIdx            = 0;
  cn->objects[objectIdx++] = objectCreate(0, 0, make_float3(0, -1, 1));
  cn->objects[objectIdx++] = objectCreate(1, 1, make_float3(0, -1, 1));
  cn->objectCount = objectIdx;
}

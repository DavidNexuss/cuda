#include <cglm/affine.h>
#include <cglm/mat4.h>
#include <cglm/vec3.h>
#include <scene.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include "cglm/cam.h"
#include "mesh.h"
#include "util.h"
#include "backend.h"
#include <cglm/cglm.h>
#include "thirdparty/lightmapper/lightmapper.h"

#define ERROR(...) dprintf(2, __VA_ARGS__)
#define VERIFY(X)                                        \
  if (!(X)) {                                            \
    dprintf(2, "Error on verifying condition " #X "\n"); \
    exit(1);                                             \
  }

#define MAX_OBJECTS 512

void GLAPIENTRY
MessageCallback(GLenum        source,
                GLenum        type,
                GLuint        id,
                GLenum        severity,
                GLsizei       length,
                const GLchar* message,
                const void*   userParam) {
  fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
          (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
          type, severity, message);
}


int   windowWidth  = 0;
int   windowHeight = 0;
float xpos;
float ypos;
float ra;

int keyboard[512];

void window_size_callback(GLFWwindow* window, int width, int height) {
  glViewport(0, 0, width, height);
  windowWidth  = width;
  windowHeight = height;

  ra = windowWidth / (float)(windowHeight);
}
void cursor_position_callback(GLFWwindow* window, double x, double y) {
  xpos = x / (float)windowWidth;
  ypos = y / (float)windowHeight;
}
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {}
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  keyboard[key] = (action == GLFW_PRESS) || (action == GLFW_REPEAT);
}

void* windowCreate(int width, int height) {

  if (glfwInit() != GLFW_TRUE) {
    ERROR("Failed to start GLFW .\n");
    return 0;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  void* window = glfwCreateWindow(width, height, "PathTracing", NULL, NULL);

  if (window == NULL) {
    ERROR("Failed to open GLFW window. width: %d height: %d\n", width, height);
    return 0;
  }

  glfwMakeContextCurrent(window);
  if (glewInit() != GLEW_OK) {
    ERROR("Failed to initialize glew");
    return 0;
  }

  glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
  glfwSetWindowSizeCallback(window, window_size_callback);
  glfwSetCursorPosCallback(window, cursor_position_callback);
  glfwSetScrollCallback(window, scroll_callback);
  glfwSetKeyCallback(window, key_callback);

  dprintf(2, "Window created with %d %d\n", width, height);
  return window;
}

void windowDestroy(GLFWwindow* window) {}

GLuint loadProgram(const char* vs, const char* fs) {

  char errorBuffer[2048];

  GLuint VertexShaderID   = glCreateShader(GL_VERTEX_SHADER);
  GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);


  GLint Result = GL_FALSE;
  int   InfoLogLength;


  const char* VertexSourcePointer   = readFile(vs);
  const char* FragmentSourcePointer = readFile(fs);

  if (VertexSourcePointer == 0) {
    ERROR("Error reading vertex shader %s \n", vs);
    return -1;
  }

  if (FragmentSourcePointer == 0) {
    ERROR("Error reading fragment shader %s \n", fs);
    return -1;
  }

  glShaderSource(VertexShaderID, 1, &VertexSourcePointer, NULL);
  glCompileShader(VertexShaderID);

  // Check Vertex Shader
  glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
  glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);

  if (InfoLogLength > 0) {
    glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, errorBuffer);
    ERROR("Error loading vertex shader %s : \n%s\n", vs, errorBuffer);
    free((void*)VertexSourcePointer);
    free((void*)FragmentSourcePointer);
    return 0;
  }


  glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, NULL);
  glCompileShader(FragmentShaderID);

  // Check Fragment Shader
  glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
  glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);

  if (InfoLogLength > 0) {
    glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, errorBuffer);
    ERROR("Error loading fragment shader %s : \n%s\n", fs, errorBuffer);
    free((void*)VertexSourcePointer);
    free((void*)FragmentSourcePointer);
    return 0;
  }

  GLuint ProgramID = glCreateProgram();
  glAttachShader(ProgramID, VertexShaderID);
  glAttachShader(ProgramID, FragmentShaderID);
  glLinkProgram(ProgramID);

  // Check the program
  glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
  glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);

  if (InfoLogLength > 0) {
    glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, errorBuffer);
    free((void*)VertexSourcePointer);
    free((void*)FragmentSourcePointer);
    return 0;
  }

  glDetachShader(ProgramID, VertexShaderID);
  glDetachShader(ProgramID, FragmentShaderID);

  glDeleteShader(VertexShaderID);
  glDeleteShader(FragmentShaderID);

  free((void*)VertexSourcePointer);
  free((void*)FragmentSourcePointer);

  return ProgramID;
}

int BUFF_PLAIN      = 0;
int BUFF_CUBE       = 1;
int BUFF_ENV        = 2;
int BUFF_START_USER = 3;

#define UNIFORMLIST(o)                     \
  o(u_envMap)                              \
    o(u_diffuseTexture)                    \
      o(u_specularTexture)                 \
        o(u_bumpTexture)                   \
          o(u_kd)                          \
            o(u_ka)                        \
              o(u_ks)                      \
                o(u_shinnness)             \
                  o(u_ro)                  \
                    o(u_rd)                \
                      o(u_isBack)          \
                        o(u_shadingMode)   \
                          o(u_useTextures) \
                            o(u_ViewMat)   \
                              o(u_ProjMat) \
                                o(u_WorldMat)

typedef struct _Renderer {
  void*        window;
  int          hintWindow;
  RendererDesc desc;

  //GL stuff
  GLuint  vao;
  GLuint* textures;
  GLuint* vbos;
  GLuint* ebos;
  GLuint* fbos;

  GLuint programPbr;

#define UNIFORM_DECL(u) GLuint pbr_##u;
  UNIFORMLIST(UNIFORM_DECL)
#undef UNIFORM_DECL

  GLuint envMap[8];

  float3 camPos;
  float3 camDir;

} Renderer;


GLfloat* indentity() {
  static GLfloat id[] = {
    1, 0, 0, 0, /*x */
    0, 1, 0, 0, /*y */
    0, 0, 1, 0, /*z*/
    0, 0, 0, 1, /*w*/
  };
  return id;
}


void setId(float* ptr) {
  for (int i = 0; i < 16; i++) ptr[i] = 0;

  ptr[0]  = 1;
  ptr[5]  = 1;
  ptr[10] = 1;
  ptr[15] = 1;
}
mat4* meshTransformPlane() {
  static mat4 plane;
  setId(&plane[0][0]);
  glm_scale(plane, (vec3){1000, 1000, 1000});
  glm_translate(plane, (vec3){-0.5, 0, -0.5});
  return &plane;
}
mat4* linearViewMatrix(float3 origin, float3 direction) {
  static float up[] = {0, 1, 0};
  static mat4  dest;
  setId(&dest[0][0]);
  glm_look((float*)&origin.x, (float*)&direction.x, up, dest);
  return &dest;
}
mat4* rendererProjMatrix(Renderer* renderer) {
  static mat4 dest;
  setId(&dest[0][0]);
  glm_perspective_default(ra, dest);
  return &dest;
}

mat4* meshTransformPlaneScreen() {
  static mat4 plane;
  float*      ptr = &plane[0][0];

  setId(&plane[0][0]);
  plane[0][0] = 2;
  plane[1][1] = 0;
  plane[2][1] = 2;
  plane[2][2] = 0;
  plane[3][0] = -1;
  plane[3][1] = -1;
  return &plane;
}

void camUpdate(float3* cam) {
  float  mag   = 0.1;
  float3 speed = make_float3((keyboard['A'] - keyboard['D']), keyboard[GLFW_KEY_SPACE] - keyboard[GLFW_KEY_LEFT_SHIFT], (keyboard['W'] - keyboard['S']));
  glm_normalize(&speed.x);

  cam->x += speed.x * mag;
  cam->y += speed.y * mag;
  cam->z += speed.z * mag;
}

void setVertexAttribs() {
  int stride = VERTEX_SIZE * sizeof(float);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, 0);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float)));
}

Renderer* rendererCreate(RendererDesc desc) {
  Renderer* renderer = (Renderer*)calloc(1, sizeof(Renderer));
  renderer->desc     = desc;
  renderer->window   = windowCreate(renderer->desc.width, renderer->desc.height);
  if (renderer->window == NULL) {
    ERROR("Failed creating window.\n");
    exit(1);
  }
  renderer->textures = (GLuint*)malloc(MAX_OBJECTS * sizeof(GLuint));
  renderer->vbos     = (GLuint*)malloc(MAX_OBJECTS * sizeof(GLuint));
  renderer->ebos     = (GLuint*)malloc(MAX_OBJECTS * sizeof(GLuint));
  renderer->fbos     = (GLuint*)malloc(MAX_OBJECTS * sizeof(GLuint));


  //GL configuration
  {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glfwWindowHint(GLFW_SAMPLES, 4);
    glEnable(GL_MULTISAMPLE);

#ifdef DEBUG
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(MessageCallback, 0);
#endif
  }
  //GL gen
  {
    glGenVertexArrays(1, &renderer->vao);
    glGenBuffers(MAX_OBJECTS, renderer->ebos);
    glGenBuffers(MAX_OBJECTS, renderer->vbos);
    glGenTextures(MAX_OBJECTS, renderer->textures);
  }

  //Upload primitive buffers
  {

    glBindBuffer(GL_ARRAY_BUFFER, renderer->vbos[BUFF_PLAIN]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, renderer->ebos[BUFF_PLAIN]);

    glBufferData(GL_ARRAY_BUFFER, sizeof(mesh_plain_vbo), mesh_plain_vbo, GL_STATIC_DRAW);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(mesh_plain_ebo), mesh_plain_ebo, GL_STATIC_DRAW);
  }

  {
    glBindVertexArray(renderer->vao);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);
  }


  renderer->programPbr = loadProgram("assets/pbr.vs", "assets/pbr.fs");

#define UNIFORM_ASSIGN(u) renderer->pbr_##u = glGetUniformLocation(renderer->programPbr, #u);
  UNIFORMLIST(UNIFORM_ASSIGN)
#undef UNIFORM_ASSIGN

  renderer->camPos = make_float3(0, 1, 0);
  renderer->camDir = make_float3(0, 0, -1);

  dprintf(2, "[Renderer] Render create completed.\n");
  return renderer;
}

void rendererInitEnvMap(Renderer* renderer, Texture* envMapTexture) {}

void rendererUpload(Renderer* renderer, Scene* scene) {
  dprintf(2, "Check %d\n", renderer->vbos[BUFF_PLAIN]);
  Texture* textureTable = (Texture*)scene->texturesTable.H;
  for (int i = 0; i < scene->textureCount; i++) {
    glActiveTexture(GL_TEXTURE0 + i);
    glBindTexture(GL_TEXTURE_2D, renderer->textures[i]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, textureTable[i].width, textureTable[i].height, 0, GL_RGB, GL_UNSIGNED_BYTE, textureTable[i].data);
    glGenerateMipmap(GL_TEXTURE_2D);
  }

  Mesh* meshList = scene->meshes.H;

  float**        vboData = (float**)scene->vertexBuffersData;
  unsigned int** eboData = (unsigned int**)scene->indexBuffersData;

  for (int i = 0; i < scene->meshCount; i++) {
    Mesh* mesh = &meshList[i];
    if (mesh->type == MESH) {
      float*        vbo = vboData[mesh->tMesh.vertexBufferIndex];
      unsigned int* ebo = eboData[mesh->tMesh.indexBufferIndex];

      dprintf(2, "[RENDERER] Uploading mesh [%d] %d %d with %d %d to %d %d\n", BUFF_START_USER + i, mesh->tMesh.vertexBufferIndex, mesh->tMesh.indexBufferIndex, mesh->tMesh.vertexCount, mesh->tMesh.indexCount, renderer->vbos[BUFF_START_USER + i], renderer->ebos[BUFF_START_USER + i]);
      glBindBuffer(GL_ARRAY_BUFFER, renderer->vbos[BUFF_START_USER + i]);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, renderer->ebos[BUFF_START_USER + i]);

      glBufferData(GL_ARRAY_BUFFER, mesh->tMesh.vertexCount * VERTEX_SIZE * sizeof(float), vbo, GL_STATIC_DRAW);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh->tMesh.indexCount * sizeof(unsigned int), ebo, GL_STATIC_DRAW);
    }
  }


  //TODO: Remove this
  SceneInput in      = sceneInputHost(scene);
  renderer->camPos   = in.constants->camera.origin;
  renderer->camPos.y = 1;

  dprintf(2, "[Renderer] Render upload completed.\n");
}
void rendererDestoy(Renderer* renderer) {

  glDeleteTextures(MAX_OBJECTS, renderer->textures);
  glDeleteBuffers(MAX_OBJECTS, renderer->vbos);
  glDeleteBuffers(MAX_OBJECTS, renderer->ebos);

  free(renderer->textures);
  free(renderer->vbos);
  free(renderer->ebos);
  free(renderer->fbos);

  windowDestroy(renderer->window);
  glfwTerminate();

  dprintf(2, "[Renderer] Render destroy completed.\n");
}

void renderBackground(Renderer* renderer) {
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glBindBuffer(GL_ARRAY_BUFFER, renderer->vbos[BUFF_PLAIN]);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, renderer->ebos[BUFF_PLAIN]);
  setVertexAttribs();
  glUniformMatrix4fv(renderer->pbr_u_WorldMat, 1, 0, (float*)*meshTransformPlaneScreen());
  glUniformMatrix4fv(renderer->pbr_u_ViewMat, 1, 0, indentity());
  glUniformMatrix4fv(renderer->pbr_u_ProjMat, 1, 0, indentity());
  glUniform1i(renderer->pbr_u_isBack, 1);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
  glUniform1i(renderer->pbr_u_isBack, 0);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
}

void bindMaterial(Renderer* renderer, Material* mat) {

  glUniform1i(renderer->pbr_u_useTextures, mat->diffuseTexture >= 0);
  if (mat->diffuseTexture >= 0) {
    glUniform1i(renderer->pbr_u_diffuseTexture, mat->diffuseTexture);
  } else {
    glUniform3f(renderer->pbr_u_kd, mat->kd.x, mat->kd.y, mat->kd.z);
    glUniform3f(renderer->pbr_u_ks, mat->ks.x, mat->ks.y, mat->ks.z);
  }
}

int bindMesh(Renderer* renderer, Mesh* mesh, int meshIdx) {
  int vertexCount = mesh->tMesh.vertexCount;
  switch (mesh->type) {
    case PLAIN:
      glBindBuffer(GL_ARRAY_BUFFER, renderer->vbos[BUFF_PLAIN]);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, renderer->ebos[BUFF_PLAIN]);
      glUniformMatrix4fv(renderer->pbr_u_WorldMat, 1, 0, (float*)*meshTransformPlane());
      vertexCount = 6;
      break;
    case MESH:
      glBindBuffer(GL_ARRAY_BUFFER, renderer->vbos[meshIdx + BUFF_START_USER]);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, renderer->ebos[meshIdx + BUFF_START_USER]);
      vertexCount = mesh->tMesh.indexCount;
      break;
  }
  return vertexCount;
}
void renderScene(Renderer* renderer, Scene* scene, float* viewMat, float* projMat, int frame) {

  SceneInput     in = sceneInputHost(scene);
  PushConstants* cn = (in.constants + frame);
  glUniformMatrix4fv(renderer->pbr_u_ViewMat, 1, 0, viewMat);
  glUniformMatrix4fv(renderer->pbr_u_ProjMat, 1, 0, projMat);
  glUniformMatrix4fv(renderer->pbr_u_WorldMat, 1, 0, indentity());

  for (int d = 0; d < cn->objectCount; d++) {
    Object* obj  = &cn->objects[d];
    Mesh*   mesh = &in.meshes[obj->mesh];

    if (obj->hasTransform) {
      glUniformMatrix4fv(renderer->pbr_u_WorldMat, 1, 0, (float*)&obj->transformMatrix->x);
    }

    int vertexCount = bindMesh(renderer, mesh, obj->mesh);
    bindMaterial(renderer, &in.materials[obj->material]);
    setVertexAttribs();
    glDrawElements(GL_TRIANGLES, vertexCount, GL_UNSIGNED_INT, 0);
  }
}
void rendererDraw(Renderer* renderer, Scene* scene) {
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glUseProgram(renderer->programPbr);
  glBindVertexArray(renderer->vao);

  SceneInput in = sceneInputHost(scene);
  for (int i = 0; i < scene->desc.framesInFlight; i++) {
    PushConstants* cn = in.constants;
    camUpdate(&renderer->camPos);

    renderer->camDir = make_float3(0, 0, 1);
    glm_normalize(&renderer->camDir.x);

    float* viewMat = (float*)*linearViewMatrix(renderer->camPos, renderer->camDir);
    float* projMat = (float*)*rendererProjMatrix(renderer);

    glUniform1i(renderer->pbr_u_envMap, cn->uniforms.skyTexture);
    glUniform3f(renderer->pbr_u_ro, viewMat[12], viewMat[13], viewMat[14]);
    glUniform3f(renderer->pbr_u_rd, viewMat[8], viewMat[9], viewMat[10]);

    renderBackground(renderer);
    renderScene(renderer, scene, viewMat, projMat, i);
  }

  glBindVertexArray(0);
}

int rendererPollEvents(Renderer* renderer) {
  glfwSwapBuffers(renderer->window);
  glfwPollEvents();
  return !glfwWindowShouldClose(renderer->window);
}

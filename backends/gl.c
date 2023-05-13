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
bool  windowMoved = 1;
float xpos;
float ypos;
float ra;
int keyboard[512];

void window_size_callback(GLFWwindow* window, int width, int height) {
  glViewport(0, 0, width, height);
  windowWidth  = width;
  windowHeight = height;

  ra = windowWidth / (float)(windowHeight);
  windowMoved = 1;
}
void cursor_position_callback(GLFWwindow* window, double x, double y) {
  xpos = x / (float)windowWidth;
  ypos = y / (float)windowHeight;

  xpos -= 0.5;
  ypos -= 0.5;

  xpos *= 2 * M_PI;
  ypos *= 2 * M_PI;
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

int TEXT_IBL = 0;
int TEXT_ATTACHMENT_COLOR = 1;
int TEXT_ATTACHMENT_BLOOM = 2;
int TEXT_GAUSS_RESULT0 = 3;
int TEXT_GAUSS_RESULT1 = 4;
int TEXT_GAUSS_RESULT2 = 5;
int TEXT_GAUSS_RESULT3 = 6;
int TEXT_START_USER = 16;

int FBO_HDR_PASS = 0;
int FBO_GAUSS_PASS = 1;
int FBO_START_USER = 2;

int RBO_HDR_PASS_DEPTH = 0;

#define UNIFORMLIST_HDR(o) \
  o(u_color) \
  o(u_bloom)

#define UNIFORMLIST_FILTER(o) o(u_input)

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
                                o(u_WorldMat) \
  o(u_flatUV)

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
  GLuint* rbos;

  GLuint programPbr;
  GLuint programGaussFilter;
  GLuint programPostHDR;

#define UNIFORM_DECL(u) GLuint pbr_##u;
  UNIFORMLIST(UNIFORM_DECL)
#undef UNIFORM_DECL

#define UNIFORM_DECL(u) GLuint hdr_##u;
  UNIFORMLIST_HDR(UNIFORM_DECL)
#undef UNIFORM_DECL

#define UNIFORM_DECL(u) GLuint filter_##u;
  UNIFORMLIST_FILTER(UNIFORM_DECL)
#undef UNIFORM_DECL

  GLuint envMap[8];

  float3 camPos;
  float3 camDir;

  bool firstFrame;

  bool engineUsingIBL;

} Renderer;


int shouldRendererRenderPass(Renderer* renderer) { 
  return renderer->desc.flag_bloom || renderer->desc.flag_bloom;
}

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
  glm_perspective(M_PI / 2.0f, ra, 0.05f, 2000.0f, dest);
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

void camUpdate(float3* cam, float3* dir) {
  float  mag   = 0.1;
  float3 speed = make_float3((keyboard['A'] - keyboard['D']), keyboard[GLFW_KEY_SPACE] - keyboard[GLFW_KEY_LEFT_SHIFT], (keyboard['W'] - keyboard['S']));
  glm_normalize(&speed.x);

  cam->x += speed.x * mag;
  cam->y += speed.y * mag;
  cam->z += speed.z * mag;

  dir->x = cos(xpos);
  dir->y = sin(ypos);
  dir->z = sin(xpos);

  glm_normalize(&dir->x);
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
  renderer->rbos     = (GLuint*)malloc(MAX_OBJECTS * sizeof(GLuint));


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
    glGenFramebuffers(MAX_OBJECTS, renderer->fbos);
    glGenRenderbuffers(MAX_OBJECTS, renderer->rbos);
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
  renderer->firstFrame = true;

  dprintf(2, "[Renderer] Render create completed.\n");
  return renderer;
}

/* Custom GL utility functions */

void glAttachScreenTexture(Renderer* renderer, int textureSlot, int attachment, GLenum type, GLenum format) { 
  if(windowMoved){ 
    glActiveTexture(GL_TEXTURE0 + textureSlot);
    glBindTexture(GL_TEXTURE_2D, renderer->textures[textureSlot]);
    glTexImage2D(GL_TEXTURE_2D, 0, format, windowWidth, windowHeight, 0 , type, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + attachment, GL_TEXTURE_2D, renderer->textures[textureSlot], 0);  
    dprintf(2, "[RENDERER] Generated screen texture for %d\n", textureSlot);
  }
}

void glAttachRenderBuffer(Renderer* renderer, int rbo, int attachment, GLenum type) { 
  if(windowMoved){ 
    glBindRenderbuffer(GL_RENDERBUFFER, renderer->rbos[rbo]);
    glRenderbufferStorage(GL_RENDERBUFFER, type, windowWidth, windowHeight);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, renderer->rbos[rbo]);  
    dprintf(2, "[RENDERER] Generated render buffer for %d\n", rbo);
  }
} 

/* End custom GL utility functions */


void rendererGenerateTextures(Renderer* renderer, Scene* scene) { 
  Texture* textureTable = (Texture*)scene->texturesTable.H;
  SceneInput in = sceneInputHost(scene);
  Texture* envMap = &textureTable[in.constants->uniforms.skyTexture];

}

void rendererUpload(Renderer* renderer, Scene* scene) {
  dprintf(2, "Check %d\n", renderer->vbos[BUFF_PLAIN]);
  Texture* textureTable = (Texture*)scene->texturesTable.H;
  for (int i = 0; i < scene->textureCount; i++) {
    glActiveTexture(GL_TEXTURE0 + i + TEXT_START_USER);
    glBindTexture(GL_TEXTURE_2D, renderer->textures[i]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, textureTable[i].width, textureTable[i].height, 0, GL_RGB, GL_UNSIGNED_BYTE, textureTable[i].data);
    glGenerateMipmap(GL_TEXTURE_2D);
  }
  
  rendererGenerateTextures(renderer, scene);
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

  dprintf(2, "[Renderer] Render upload completed.\n");
}
void rendererUploadObjects(Renderer* renderer, Scene* scene) { 
  if(renderer->firstFrame) {
    SceneInput in      = sceneInputHost(scene);
    renderer->camPos   = in.constants->camera.origin;
    renderer->camPos.y = 1;
    renderer->firstFrame = 0;
  }
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

void rendererRenderScreenMesh(Renderer* renderer, GLuint worldMat, GLuint viewMat, GLuint projMat) { 
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glBindBuffer(GL_ARRAY_BUFFER, renderer->vbos[BUFF_PLAIN]);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, renderer->ebos[BUFF_PLAIN]);
  setVertexAttribs();
  glUniformMatrix4fv(worldMat, 1, 0, (float*)*meshTransformPlaneScreen());
  glUniformMatrix4fv(viewMat, 1, 0, indentity());
  glUniformMatrix4fv(projMat, 1, 0, indentity());
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
}

void rendererRenderScreenQuad(Renderer* renderer) { 
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glDrawArrays(GL_TRIANGLES, 6, 0);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
}

void bindMaterial(Renderer* renderer, Material* mat) {

  glUniform1i(renderer->pbr_u_useTextures, mat->diffuseTexture >= 0);
  if (mat->diffuseTexture >= 0) {
    glUniform1i(renderer->pbr_u_diffuseTexture, mat->diffuseTexture + TEXT_START_USER);
  } else {
    glUniform3f(renderer->pbr_u_kd, mat->kd.x, mat->kd.y, mat->kd.z);
    glUniform3f(renderer->pbr_u_ks, mat->ks.x, mat->ks.y, mat->ks.z);
  }
}

int bindMesh(Renderer* renderer, Mesh* mesh, int meshIdx) {
  int vertexCount = mesh->tMesh.vertexCount;
  glUniform1i(renderer->pbr_u_flatUV, 0);
  switch (mesh->type) {
    case PLAIN:
      glBindBuffer(GL_ARRAY_BUFFER, renderer->vbos[BUFF_PLAIN]);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, renderer->ebos[BUFF_PLAIN]);
      glUniformMatrix4fv(renderer->pbr_u_WorldMat, 1, 0, (float*)*meshTransformPlane());
      glUniform1i(renderer->pbr_u_flatUV, 1);
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

  for (int d = 0; d < cn->objectCount; d++) {
    Object* obj  = &cn->objects[d];
    Mesh*   mesh = &in.meshes[obj->mesh];

    if (obj->hasTransform) {
      glUniformMatrix4fv(renderer->pbr_u_WorldMat, 1, 0, (float*)&obj->transformMatrix->x);
    } else { 
      glUniformMatrix4fv(renderer->pbr_u_WorldMat, 1, 0, indentity());
    }

    int vertexCount = bindMesh(renderer, mesh, obj->mesh);
    bindMaterial(renderer, &in.materials[obj->material]);
    setVertexAttribs();
    glDrawElements(GL_TRIANGLES, vertexCount, GL_UNSIGNED_INT, 0);
  }
}

void rendererPass(Renderer* renderer, Scene* scene) {
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glUseProgram(renderer->programPbr);

  SceneInput in = sceneInputHost(scene);
  for (int i = 0; i < scene->desc.framesInFlight; i++) {
    PushConstants* cn = in.constants;
    camUpdate(&renderer->camPos, &renderer->camDir);
    glm_normalize(&renderer->camDir.x);

    float* viewMat = (float*)*linearViewMatrix(renderer->camPos, renderer->camDir);
    float* projMat = (float*)*rendererProjMatrix(renderer);

    glUniform1i(renderer->pbr_u_envMap, cn->uniforms.skyTexture + TEXT_START_USER);
    glUniform3f(renderer->pbr_u_ro, renderer->camPos.x, renderer->camPos.y, renderer->camPos.z);
    glUniform3f(renderer->pbr_u_rd, renderer->camDir.x, renderer->camDir.y, renderer->camDir.z);

    glUniform1i(renderer->pbr_u_isBack, 1);
    rendererRenderScreenMesh(renderer, renderer->pbr_u_WorldMat, renderer->pbr_u_ViewMat, renderer->pbr_u_ProjMat);
    glUniform1i(renderer->pbr_u_isBack, 0);
    renderScene(renderer, scene, viewMat, projMat, i);
  }
}


void rendererBeginHDR(Renderer* renderer) { 
  glBindFramebuffer(GL_FRAMEBUFFER, renderer->fbos[FBO_HDR_PASS]);
  glAttachRenderBuffer(renderer, RBO_HDR_PASS_DEPTH, GL_DEPTH_STENCIL_ATTACHMENT, GL_DEPTH24_STENCIL8);
  glAttachScreenTexture(renderer, TEXT_ATTACHMENT_COLOR, 0, GL_RGB, GL_RGB16F);
  if(renderer->desc.flag_bloom) { 
     glAttachScreenTexture(renderer, TEXT_ATTACHMENT_BLOOM, 1, GL_RGB, GL_RGB16F);
  }
}

void rendererFilterGauss(Renderer* renderer, int input, int output) { 
  glBindFramebuffer(GL_FRAMEBUFFER, renderer->fbos[FBO_GAUSS_PASS]);
  glAttachScreenTexture(renderer, output, 0, GL_RGB, GL_RGB16F);
  rendererRenderScreenQuad(renderer);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void rendererEnd() { 
  glBindFramebuffer(GL_FRAMEBUFFER,0);
}

void rendererDraw(Renderer* renderer, Scene* scene) {

  glBindVertexArray(renderer->vao);
  int doHdrPass = shouldRendererRenderPass(renderer);
  if(doHdrPass) rendererBeginHDR(renderer);
  rendererPass(renderer, scene);
  if(doHdrPass) { 
    rendererEnd();
    if(renderer->desc.flag_bloom) { 
      rendererFilterGauss(renderer, TEXT_ATTACHMENT_BLOOM, TEXT_GAUSS_RESULT0);
    }

  }
  windowMoved = 0;
  glBindVertexArray(0);
}

int rendererPollEvents(Renderer* renderer) {
  glfwSwapBuffers(renderer->window);
  glfwPollEvents();
  return !glfwWindowShouldClose(renderer->window);
}

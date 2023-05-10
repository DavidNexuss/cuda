#include <scene.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include "mesh.h"
#include "util.h"
#include "backend.h"

#define ERROR(...) dprintf(2, __VA_ARGS__)

#define MAX_OBJECTS 512

void window_size_callback(GLFWwindow* window, int width, int height) { glViewport(0, 0, width, height); }
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {}
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {}
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {}

void* windowCreate(int width, int height) {

  if (glfwInit() != GLFW_TRUE) {
    ERROR("Failed to start GLFW .\n");
    return 0;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
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

  char errorBuffer[512];

  GLuint VertexShaderID   = glCreateShader(GL_VERTEX_SHADER);
  GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);


  GLint Result = GL_FALSE;
  int   InfoLogLength;


  const char* VertexSourcePointer   = readFile(vs);
  const char* FragmentSourcePointer = readFile(fs);

  if (VertexSourcePointer == 0) {
    ERROR("Error reading vertex shader");
    return -1;
  }

  if (FragmentSourcePointer == 0) {
    ERROR("Error reading fragment shader");
    return -1;
  }

  glShaderSource(VertexShaderID, 1, &VertexSourcePointer, NULL);
  glCompileShader(VertexShaderID);

  // Check Vertex Shader
  glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
  glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);

  if (InfoLogLength > 0) {
    glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, errorBuffer);
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

typedef struct _Renderer {
  void*        window;
  int          hintWindow;
  RendererDesc desc;

  //GL stuff
  GLuint* textures;
  GLuint* vbos;
  GLuint* fbos;

  int numTextures;
  int numVbos;
  int numPbos;

  GLuint vboPlain;
  GLuint vboCube;
  GLuint vboEnv;

  GLuint objectVao;

  GLuint programPbr;
  GLuint programPbrCamera;
  GLuint programPbrViewMat;
  GLuint programPbrProjMat;

} Renderer;


GLfloat* linearViewMatrix(float3 origin, float3 direction) {}
GLfloat* rendererProjMatrix(Renderer* renderer) {}

Renderer* rendererCreate(RendererDesc desc) {
  Renderer* renderer = (Renderer*)calloc(1, sizeof(Renderer));
  renderer->desc     = desc;
  renderer->window   = windowCreate(renderer->desc.width, renderer->desc.height);
  if (renderer->window == NULL) {
    ERROR("Failed creating window.\n");
    exit(1);
  }
  renderer->textures = (GLuint*)calloc(MAX_OBJECTS, sizeof(GLuint*));
  renderer->vbos     = (GLuint*)calloc(MAX_OBJECTS, sizeof(GLuint*));
  renderer->fbos     = (GLuint*)calloc(MAX_OBJECTS, sizeof(GLuint*));

  glGenBuffers(1, &renderer->vboPlain);
  glGenBuffers(1, &renderer->vboCube);
  glGenBuffers(1, &renderer->vboEnv);
  glGenVertexArrays(1, &renderer->objectVao);

  glBindVertexArray(renderer->objectVao);
  glBufferData(GL_ARRAY_BUFFER, sizeof(mesh_plain), mesh_plain, GL_STATIC_DRAW);

  renderer->programPbr = loadProgram("assets/pbr.vs", "assets/pbr.fs");
  return renderer;
}

void rendererUpload(Renderer* renderer, Scene* scene) {
  renderer->numTextures = scene->textureCount;
  glGenTextures(scene->textureCount, renderer->textures);
  Texture* textureTable = (Texture*)scene->texturesTable.H;
  for (int i = 0; i < scene->textureCount; i++) {
    glBindTexture(GL_TEXTURE_2D, renderer->textures[i]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, textureTable[i].width, textureTable[i].height, 0, GL_RGB, GL_UNSIGNED_BYTE, textureTable[i].data);
  }
}
void rendererDestoy(Renderer* renderer) {

  for (int i = 0; i < renderer->numTextures; i++) {
    glDeleteTextures(renderer->numTextures, renderer->textures);
  }

  free(renderer->textures);
  free(renderer->vbos);
  free(renderer->fbos);

  glfwTerminate();
  windowDestroy(renderer->window);
}

void rendererDraw(Renderer* renderer, Scene* scene) {
  glUseProgram(renderer->programPbr);

  SceneInput in = sceneInputHost(scene);
  for (int i = 0; i < scene->desc.framesInFlight; i++) {

    PushConstants* cn = in.constants;
    glUniformMatrix4fv(renderer->programPbrViewMat, 1, 0, linearViewMatrix(cn->camera.origin, cn->camera.direction));
    glUniformMatrix4fv(renderer->programPbrProjMat, 1, 0, rendererProjMatrix(renderer));
    for (int d = 0; d < cn->objectCount; d++) {
    }
  }
}

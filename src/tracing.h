#include "objects.h"
#include "scene.h"
__global__ void pathTracingKernel(int width, int height, float* fbo_mat, int iterationsPerThread, int maxDepth, SceneInput input);

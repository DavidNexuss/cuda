#include <math.h>
static float mesh_plain[] = {
  /*POSITION*/ 0, 0, 0, /*NORMAL*/ 0, 1, 0,
  /*POSITION*/ 0, 0, 1, /*NORMAL*/ 0, 1, 0,
  /*POSITION*/ 1, 0, 1, /*NORMAL*/ 0, 1, 0,
  /*POSITION*/ 1, 0, 1, /*NORMAL*/ 0, 1, 0,
  /*POSITION*/ 1, 0, 0, /*NORMAL*/ 0, 1, 0,
  /*POSITION*/ 0, 0, 0, /*NORMAL*/ 0, 1, 0};

float* mesh_uv_sphere(int rings, int segments) {

  for (int i = 0; i < rings; i++) {
    float y = sin(i);
    for (int j = 0; j < segments; j++) {
      float x = cos(j);
      float z = sin(j);
    }
  }

  return 0;
}

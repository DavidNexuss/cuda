#include <math.h>
static float mesh_plain_vbo[] = {
  /*POSITION*/ 0, 0, 0, /*NORMAL*/ 0, 1, 0, /* UV */ 0, 0,
  /*POSITION*/ 0, 0, 1, /*NORMAL*/ 0, 1, 0, /* UV */ 0, 1,
  /*POSITION*/ 1, 0, 1, /*NORMAL*/ 0, 1, 0, /* UV */ 1, 1,
  /*POSITION*/ 1, 0, 0, /*NORMAL*/ 0, 1, 0, /* UV */ 1, 0};

static unsigned int mesh_plain_ebo[] = {0, 1, 2, 3, 0, 2};

void mesh_uv_sphere(int rings, int segments, float** vbo_out, unsigned short** ebo_out) {

  int    s   = ((rings - 2) * segments + 2) * 3;
  float* vbo = (float*)malloc(sizeof(float) * s);

  vbo[0] = 0;
  vbo[1] = 0;
  vbo[2] = 0;

  vbo[s - 3] = 0;
  vbo[s - 2] = 1;
  vbo[s - 1] = 0;

  int t = 0;
  for (int i = 1; i < rings - 1; i++) {
    float y = cos(M_PI * i / (float)rings);
    for (int j = 0; j < segments; j++) {
      float x = cos(2 * M_PI * j / (float)segments);
      float z = sin(2 * M_PI * j / (float)segments);

      vbo[t]     = x;
      vbo[t + 1] = y;
      vbo[t + 2] = z;
      t += 3;
    }
  }

  *vbo_out = vbo;
}

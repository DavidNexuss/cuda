#pragma once
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <malloc.h>

const char* readFile(const char* path) {
  struct stat _stat;
  stat(path, &_stat);

  if (_stat.st_size <= 0) return 0;

  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    return 0;
  }

  char* buffer          = (char*)malloc(_stat.st_size + 1);
  buffer[_stat.st_size] = 0;

  int current = 0;
  int size    = _stat.st_size;
  int step    = 0;

  while ((step = read(fd, &buffer[current], size - current))) {
    current += step;
  }

  return buffer;
}

// Chat gpt generated function
// Function to apply the Gaussian blur to a given pixel
void apply_gaussian_blur(float* input, float* output, int width, int height, int x, int y, int kernelSize, float sigma) {
  const int NUM_CHANNELS = 3;
  // Compute the center of the kernel
  int center = kernelSize / 2;

  // Compute the sum of the weights in the kernel
  float weight_sum = 0.0f;
  for (int i = 0; i < kernelSize; i++) {
    for (int j = 0; j < kernelSize; j++) {
      float weight = expf(-((i - center) * (i - center) + (j - center) * (j - center)) / (2 * sigma * sigma));
      weight_sum += weight;
    }
  }

  // Compute the weighted average of the colors in the kernel
  float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
  for (int i = 0; i < kernelSize; i++) {
    for (int j = 0; j < kernelSize; j++) {
      // Compute the index of the current pixel in the input array
      int index = ((y + i - center) * width + (x + j - center)) * NUM_CHANNELS;

      // Get the color values for the current pixel
      float r = input[index];
      float g = input[index + 1];
      float b = input[index + 2];

      // Compute the weight for the current pixel
      float weight = expf(-((i - center) * (i - center) + (j - center) * (j - center)) / (2 * sigma * sigma));

      // Accumulate the weighted color values
      r_sum += weight * r;
      g_sum += weight * g;
      b_sum += weight * b;
    }
  }

  // Normalize the weighted color values by the sum of the weights
  output[(y * width + x) * NUM_CHANNELS]     = r_sum / weight_sum;
  output[(y * width + x) * NUM_CHANNELS + 1] = g_sum / weight_sum;
  output[(y * width + x) * NUM_CHANNELS + 2] = b_sum / weight_sum;
}

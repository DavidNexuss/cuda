#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

const char* readFile(const char* path) {
  struct stat _stat;
  stat(path, &_stat);

  if (_stat.st_size <= 0) return 0;

  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    return 0;
  }

  char* buffer          = malloc(_stat.st_size + 1);
  buffer[_stat.st_size] = 0;

  int current = 0;
  int size    = _stat.st_size;
  int step    = 0;

  while ((step = read(fd, &buffer[current], size - current))) {
    current += step;
  }

  return buffer;
}

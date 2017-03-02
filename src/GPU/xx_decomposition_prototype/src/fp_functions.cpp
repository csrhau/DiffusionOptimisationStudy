#include "fp_functions.h"

#include <math.h>

int FloatCompare(float a, float b) {
  const float eps = 1E-8f;
  const float delta = a - b;
  if (fabs(delta) < eps) { 
    return 0;
  } else if (a < b) {
    return -1;
  } else {
    return 1;
  }
}

float ParseFloat(char *str) {
  errno = 0;
  char *endptr;
  float val = strtof(str, &endptr);
  if ((errno == ERANGE && (val == LONG_MAX || val == LONG_MIN))
       || (errno != 0 && val == 0)) {
    perror("strtof");
    exit(EXIT_FAILURE);
  }
  if (endptr == str) {
    fprintf(stderr, "No digits were found\n");
    exit(EXIT_FAILURE);
  }
  return val;
}

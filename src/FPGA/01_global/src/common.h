#pragma once

#define TIMESTEPS 5

#define IMAX 32
#define JMAX 32
#define KMAX 32

#define HOTCORNER_IMAX 25
#define HOTCORNER_JMAX 25
#define HOTCORNER_KMAX 25

#define INDEX3D(i, j, k) ((((k) * (JMAX) * (IMAX)) + ((j) * (IMAX)) + (i)))

enum Registers {
  TEMPERATURE = 0, // First entry has to start at zero
  REGISTER_COUNT,  // Must be last entry!
};

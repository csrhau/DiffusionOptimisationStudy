#pragma once 
#define TIMESTEPS 100
#define GRINDTIME 10

// 250^3 is the absolute limit of correctness
#define IMAX 250
#define JMAX 250
#define KMAX 250

#define ISTRIPWIDTH 8
#define JSTRIPWIDTH 8
#define KSTRIPWIDTH 8

#define HOTCORNER_IMAX 25
#define HOTCORNER_JMAX 25
#define HOTCORNER_KMAX 25

#define INDEX3D(i, j, k) ((((k) * (JMAX) * (IMAX)) + ((j) * (IMAX)) + (i)))

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

enum Registers {
  TEMPERATURE = 0, // First entry has to start at zero
  REGISTER_COUNT,  // Must be last entry!
};

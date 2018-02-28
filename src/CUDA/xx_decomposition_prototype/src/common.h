#pragma once
#define TIMESTEPS 5

#define STENCIL_RADIUS 1

#define IMAX 32
#define JMAX 32
#define KMAX 32

#define HOTCORNER_IMAX 25
#define HOTCORNER_JMAX 25
#define HOTCORNER_KMAX 25

#define INDEX3D(i, j, k) ((((k) * (JMAX) * (IMAX)) + ((j) * (IMAX)) + (i)))

#define T_THRESHOLD 8
#define I_THRESHOLD 16
#define J_THRESHOLD 16
#define K_THRESHOLD 16

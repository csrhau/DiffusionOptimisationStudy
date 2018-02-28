#pragma once
/* vim: set ft=cpp */

#define TIMESTEPS 600

#define IMAX 800
#define JMAX 800
#define KMAX 800

#define HOTCORNER_IMAX 25
#define HOTCORNER_JMAX 25
#define HOTCORNER_KMAX 25

#define INDEX3D(i, j, k) ((((k) * (JMAX) * (IMAX)) + ((j) * (IMAX)) + (i)))

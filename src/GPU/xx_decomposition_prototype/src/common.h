#define TIMESTEPS 60

#define STENCIL_RADIUS 1

#define IMAX 30
#define JMAX 30
#define KMAX 30

#define HOTCORNER_IMAX 25
#define HOTCORNER_JMAX 25
#define HOTCORNER_KMAX 25

#define INDEX3D(i, j, k) ((((k) * (JMAX) * (IMAX)) + ((j) * (IMAX)) + (i)))

#define T_THRESHOLD 8
#define I_THRESHOLD 8
#define J_THRESHOLD 8
#define K_THRESHOLD 8

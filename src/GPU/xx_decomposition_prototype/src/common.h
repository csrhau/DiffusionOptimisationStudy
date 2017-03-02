#define TIMESTEPS 60

#define STENCIL_RADIUS 1

#define IMAX 50
#define JMAX 50
#define KMAX 50

#define HOTCORNER_IMAX 25
#define HOTCORNER_JMAX 25
#define HOTCORNER_KMAX 25

#define INDEX3D(i, j, k) ((((k) * (JMAX) * (IMAX)) + ((j) * (IMAX)) + (i)))

#define T_THRESHOLD 20
#define I_THRESHOLD 10
#define J_THRESHOLD 10
#define K_THRESHOLD 10

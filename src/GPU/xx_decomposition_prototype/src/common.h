#define TIMESTEPS 100

#define IMAX 400
#define JMAX 400
#define KMAX 400

#define HOTCORNER_IMAX 25
#define HOTCORNER_JMAX 25
#define HOTCORNER_KMAX 25

#define INDEX3D(i, j, k) ((((k) * (JMAX) * (IMAX)) + ((j) * (IMAX)) + (i)))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define T_THRESHOLD 5
#define I_THRESHOLD 16
#define J_THRESHOLD 16
#define K_THRESHOLD 16

#define STENCIL_RADIUS 1

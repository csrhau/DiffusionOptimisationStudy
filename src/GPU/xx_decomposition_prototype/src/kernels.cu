#include "kernels.cuh"

#include "common.h"


void BaseTrapezoid(double *state[2],
                   double cx, double cy, double cz,
                   int t0, int t1,
                   int i0, int del_i0, int i1, int del_i1, int imax,
                   int j0, int del_j0, int j1, int del_j1, int jmax,
                   int k0, int del_k0, int k1, int del_k1, int kmax) {
  for (int ts = t0; ts < t1; ++ts) {
    double *__restrict__ state_now = state[ts & 1];
    double *__restrict__ state_next = state[(ts + 1) & 1];
    for (int k = k0; k < k1; ++k) {
      for (int j = j0; j < j1; ++j) {
        for (int i = i0; i < i1; ++i) {
          const size_t center = INDEX3D(i, j, k);
          // Diffusion
          state_next[center] = state_now[center] + cx * (state_now[center-1]-2.0* state_now[center] + state_now[center+1])
                                                 + cy * (state_now[center-imax]-2.0* state_now[center] + state_now[center+imax])
                                                 + cz * (state_now[center-imax*jmax]-2.0* state_now[center] + state_now[center+imax*jmax]);
          // Reflective Boundary Conditions
          if (i==1) {
            state_next[INDEX3D(i-1, j, k)] = state_next[INDEX3D(i, j, k)];
          } else if (i == imax - 2) {
            state_next[INDEX3D(i+1, j, k)] = state_next[INDEX3D(i, j, k)];
          } 
          if (j == 1) {
            state_next[INDEX3D(i, j-1, k)] = state_next[INDEX3D(i, j, k)];
          } else if (j == jmax - 2) {
            state_next[INDEX3D(i, j+1, k)] = state_next[INDEX3D(i, j, k)];
          } 
          if (k == 1) {
            state_next[INDEX3D(i, j, k-1)] = state_next[INDEX3D(i, j, k)];
          } else if (k == kmax - 2) {
            state_next[INDEX3D(i, j, k+1)] = state_next[INDEX3D(i, j, k)];
          } 
        }
      }
    }
    i0 += del_i0;
    i1 += del_i1;
    j0 += del_j0;
    j1 += del_j1;
    k0 += del_k0;
    k1 += del_k1;
  }
}

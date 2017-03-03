#include "kernels.cuh"

#include <stdio.h>
#include "common.h"

__global__ void DiffuseReflectKnl(
    double *__restrict__ state_now,
    double *__restrict__ state_next,
    int ts,
    int i0, int i1, int imax,
    int j0, int j1, int jmax,
    int k0, int k1, int kmax,
    double cx,
    double cy,
    double cz) {
  // All indexing based on 1D offset
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int ispan = i1 - i0;
  const int jspan = j1 - j0;
  const int i = i0 + (tid % ispan);
  const int j = j0 + ((tid / ispan) % jspan);
  const int k = k0 + (tid / (ispan * jspan));
  if (i >= i0 && i <= i1 && j >= j0 && j <= j1 && k >= k0 && k <= k1) {
     // Diffuse
    const int center = INDEX3D(i, j, k);
    state_next[center] = state_now[center] + cx * (state_now[center-1]-2.0*state_now[center]+state_now[center+1])
                                           + cy * (state_now[center-imax]-2.0*state_now[center]+state_now[center+imax])
                                           + cz * (state_now[center-imax*jmax]-2.0*state_now[center]+state_now[center+imax*jmax]);
    // Reflect
    if (i == 1) {
      state_next[center-1] = state_next[center];
    } else if (i == imax - 2) {
      state_next[center+1] = state_next[center];
    }
    if (j == 1) {
      state_next[center-imax] = state_next[center];
    } else if (j == jmax - 2) {
      state_next[center+imax] = state_next[center];
    }
    if (k == 1) {
      state_next[center-imax*jmax] = state_next[center];
    } else if (k == kmax - 2) {
      state_next[center+imax*jmax] = state_next[center];
    }
  }
}

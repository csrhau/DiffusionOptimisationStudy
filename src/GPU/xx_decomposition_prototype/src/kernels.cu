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
  const int kspan = k1 - k0;
  int residual = tid;
  int kl = tid / (ispan * jspan);
  residual -= kl * ispan * jspan;
  int jl = residual / ispan;
  residual -= jl * ispan;
  int il = residual;
  if (il < ispan && jl < jspan && kl < kspan) {
    int i = il + i0;
    int j = jl + j0;
    int k = kl + k0;
    // Diffusion
    const size_t center = INDEX3D(i, j, k);
    state_next[center] = state_now[center] + 1;/* + cx * (state_now[center-1]-2.0* state_now[center] + state_now[center+1])
                                           + cy * (state_now[center-imax]-2.0* state_now[center] + state_now[center+imax])
                                           + cz * (state_now[center-imax*jmax]-2.0* state_now[center] + state_now[center+imax*jmax]);
                                           */
    // Reflective Boundary Conditions
    if (i==1) {
      state_next[INDEX3D(i-1, j, k)] = state_next[INDEX3D(i, j, k)];
    } else if (i == IMAX - 2) {
      state_next[INDEX3D(i+1, j, k)] = state_next[INDEX3D(i, j, k)];
    } 
    if (j == 1) {
      state_next[INDEX3D(i, j-1, k)] = state_next[INDEX3D(i, j, k)];
    } else if (j == JMAX - 2) {
      state_next[INDEX3D(i, j+1, k)] = state_next[INDEX3D(i, j, k)];
    } 
    if (k == 1) {
      state_next[INDEX3D(i, j, k-1)] = state_next[INDEX3D(i, j, k)];
    } else if (k == KMAX - 2) {
      state_next[INDEX3D(i, j, k+1)] = state_next[INDEX3D(i, j, k)];
    } 
  }
}


__global__ void PrintKnl(
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
  const int kspan = k1 - k0;
  printf("%d THREAD ispan %d jspan %d kspan %d\n", tid, ispan, jspan, kspan);
  int residual = tid;
  int kl = tid / (ispan * jspan);
  residual -= kl * ispan * jspan;
  int jl = residual / ispan;
  residual -= jl * ispan;
  int il = residual;
  int i = il + i0;
  int j = jl + j0;
  int k = kl + k0;
  if (il < ispan && jl < jspan && kl < kspan) {
    printf("%d %d %d %d (t,i,j,k) DEVICE\n", ts, i, j, k);
  }
}

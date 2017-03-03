#include "dispatch.cuh"

#include "common.h"

#include "kernels.cuh"

void LaunchSingleIteration(double *__restrict__ state_now,
                           double *__restrict__ state_next,
                           int ts,
                           int i0, int i1, int imax,
                           int j0, int j1, int jmax,
                           int k0, int k1, int kmax,
                           double cx,
                           double cy,
                           double cz) {
  double *device_now, *device_next;
  cudaMalloc((void **) &device_now, imax * jmax * kmax * sizeof(double));
  cudaMalloc((void **) &device_next, imax * jmax * kmax * sizeof(double));
  cudaMemcpy(device_now, state_now, imax * jmax * kmax * sizeof(double), cudaMemcpyHostToDevice);
  int tpb = (imax * jmax * kmax + 127) / 128;
  DiffuseReflectKnl<<<tpb, 128>>>(device_now, device_next,
                                  ts,
                                  i0, i1, imax,
                                  j0, j1, jmax,
                                  k0, k1, kmax,
                                  cx, cy, cz);
  // Run single iteration
  cudaMemcpy(state_next, device_next, imax * jmax * kmax * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(device_now);
  cudaFree(device_next);
}

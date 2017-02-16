#pragma once
/* vim: set ft=cpp : */ 

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "common.cuh"


template <int ir, int jr, int kr>
__global__ void DiffuseKnl(
    double *__restrict__ tnow,
    double *__restrict__ tnext,
    double cx,
    double cy,
    double cz) {
  for (int is = 0; is < ir; ++is) {
    const int i = gridDim.x * blockDim.x * is + blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i - 1 < IMAX - 2) {
      for (int js = 0; js < jr; ++js) {
        const int j = gridDim.y * blockDim.y * js + blockIdx.y * blockDim.y + threadIdx.y + 1;
        if (j - 1 < JMAX - 2) {
          for (int ks = 0; ks < kr; ++ks) {
            const int k = gridDim.z * blockDim.z * ks + blockIdx.z * blockDim.z + threadIdx.z + 1;
            if (k - 1 < KMAX - 2) {
              // Diffuse
              tnext[INDEX3D(i, j, k)] = tnow[INDEX3D(i, j, k)] + cx * (tnow[INDEX3D(i-1, j, k)] - 2.0*tnow[INDEX3D(i, j, k)] + tnow[INDEX3D(i+1, j, k)])
                                                               + cy * (tnow[INDEX3D(i, j-1, k)] - 2.0*tnow[INDEX3D(i, j, k)] + tnow[INDEX3D(i, j+1, k)])
                                                               + cz * (tnow[INDEX3D(i, j, k-1)] - 2.0*tnow[INDEX3D(i, j, k)] + tnow[INDEX3D(i, j, k+1)]);

            }
          }
        }
      }
    }
  }
}


__global__ void ReflectIKnl(double *tnext);
__global__ void ReflectJKnl(double *tnext);
__global__ void ReflectKKnl(double *tnext);


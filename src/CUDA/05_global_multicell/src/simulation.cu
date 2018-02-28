#include "simulation.cuh"

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "common.cuh"


__global__ void ReflectIKnl(double *__restrict__ tnext) {
  const int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
  if (j - 1 < JMAX - 2 && k - 1 < KMAX - 2) {
    tnext[INDEX3D(0, j, k)] = tnext[INDEX3D(1, j, k)];
    tnext[INDEX3D(IMAX-1, j, k)] = tnext[INDEX3D(IMAX-2, j, k)];
  }
}

__global__ void ReflectJKnl(double *__restrict__ tnext) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
  if (i - 1 < IMAX - 2 && k - 1 < KMAX - 2) {
    tnext[INDEX3D(i, 0, k)] = tnext[INDEX3D(i, 1, k)];
    tnext[INDEX3D(i, JMAX-1, k)] = tnext[INDEX3D(i, JMAX-2, k)];
  }
}

__global__ void ReflectKKnl(double *__restrict__ tnext) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i - 1 < IMAX - 2 && j - 1 < JMAX - 2) {
    tnext[INDEX3D(i, j, 0)] = tnext[INDEX3D(i, j, 1)];
    tnext[INDEX3D(i, j, KMAX-1)] = tnext[INDEX3D(i, j, KMAX-2)];
  }
}

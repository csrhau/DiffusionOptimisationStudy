#include "simulation.cuh"

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "common.cuh"

__global__ void DiffuseReflectKnl(
    double *__restrict__ tnow,
    double *__restrict__ tnext,
    double cx,
    double cy,
    double cz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
  if (i - 1 < IMAX - 2 && j - 1 < JMAX - 2 && k - 1 < KMAX - 2) {
     // Diffuse
    tnext[INDEX3D(i, j, k)] = tnow[INDEX3D(i, j, k)] + cx * (tnow[INDEX3D(i-1, j, k)] - 2.0*tnow[INDEX3D(i, j, k)] + tnow[INDEX3D(i+1, j, k)])
                                                     + cy * (tnow[INDEX3D(i, j-1, k)] - 2.0*tnow[INDEX3D(i, j, k)] + tnow[INDEX3D(i, j+1, k)])
                                                     + cz * (tnow[INDEX3D(i, j, k-1)] - 2.0*tnow[INDEX3D(i, j, k)] + tnow[INDEX3D(i, j, k+1)]);
    // Reflect
    if (i == 1) {
      tnext[INDEX3D(0, j, k)] = tnext[INDEX3D(i, j, k)];
    } else if (i == IMAX - 2) {
      tnext[INDEX3D(IMAX-1, j, k)] = tnext[INDEX3D(i, j, k)];
    }
    if (j == 1) {
      tnext[INDEX3D(i, 0, k)] = tnext[INDEX3D(i, j, k)];
    } else if (j == JMAX - 2) {
      tnext[INDEX3D(i, JMAX-1, k)] = tnext[INDEX3D(i, j, k)];
    }
    if (k == 1) {
      tnext[INDEX3D(i, j, 0)] = tnext[INDEX3D(i, j, k)];
    } else if (k == KMAX - 2) {
      tnext[INDEX3D(i, j, KMAX-1)] = tnext[INDEX3D(i, j, k)];
    }
  }
}

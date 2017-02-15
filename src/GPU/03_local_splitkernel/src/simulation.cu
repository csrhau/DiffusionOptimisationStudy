#include "simulation.cuh"

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "common.cuh"

__global__ void DiffuseKnl(
    double *__restrict__ tnow,
    double *__restrict__ tnext,
    double cx,
    double cy,
    double cz) {
  extern __shared__ double sdata[];
  const int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
  if (i - 1 < IMAX - 2 && j - 1 < JMAX - 2 && k - 1 < KMAX - 2) {
    const int jstride = blockDim.x + 2;
    const int kstride = (blockDim.x + 2) * (blockDim.y + 2);
    const int il = threadIdx.x + 1;
    const int jl = threadIdx.y + 1;
    const int kl = threadIdx.z + 1;
    const int center = kl * kstride + jl * jstride + il;
    sdata[center] = tnow[INDEX3D(i, j, k)];
    if (threadIdx.x == 0) {
      sdata[center-1] = tnow[INDEX3D(i-1, j, k)];
    }
    if (threadIdx.x == blockDim.x-1 || i == IMAX-2) {
      sdata[center+1] = tnow[INDEX3D(i+1, j, k)];
    }
    if (threadIdx.y == 0) {
      sdata[center-jstride] = tnow[INDEX3D(i, j-1, k)];
    } 
    if (threadIdx.y == blockDim.y-1 || j == JMAX-2) {
      sdata[center+jstride] = tnow[INDEX3D(i, j+1, k)];
    }
    if (threadIdx.z == 0) {
      sdata[center-kstride] = tnow[INDEX3D(i, j, k-1)];
    } 
    if (threadIdx.z == blockDim.z-1 || k == KMAX-2) {
      sdata[center+kstride] = tnow[INDEX3D(i, j, k+1)];
    }
    __syncthreads();
     // Diffuse
    tnext[INDEX3D(i, j, k)] = sdata[center] + cx * (sdata[center-1] - 2.0*sdata[center] + sdata[center+1])
                                            + cy * (sdata[center-jstride] - 2.0*sdata[center] + sdata[center+jstride])
                                            + cz * (sdata[center-kstride]- 2.0*sdata[center] + sdata[center+kstride]);
  }
}

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

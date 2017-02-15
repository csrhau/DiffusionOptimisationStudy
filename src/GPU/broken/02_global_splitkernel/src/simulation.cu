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
  const int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
  if (i - 1 < IMAX - 2 && j - 1 < JMAX - 2 && k - 1 < KMAX - 2) {
     // Diffuse
    tnext[INDEX3D(i, j, k)] = tnow[INDEX3D(i, j, k)] + cx * (tnow[INDEX3D(i-1, j, k)] - 2.0*tnow[INDEX3D(i, j, k)] + tnow[INDEX3D(i+1, j, k)])
                                                     + cy * (tnow[INDEX3D(i, j-1, k)] - 2.0*tnow[INDEX3D(i, j, k)] + tnow[INDEX3D(i, j+1, k)])
                                                     + cz * (tnow[INDEX3D(i, j, k-1)] - 2.0*tnow[INDEX3D(i, j, k)] + tnow[INDEX3D(i, j, k+1)]);
  }
}

// Note, i and j here are not tied to the 3d coordinate system
// They are offsets into a 2d plane, which can be oriented normal 
// to any of the 3d i,j,k axis (so 2d i, j can represent any 3d axis)
__global__ void ReflectKnl(double *__restrict__ tnext,
                           int offset, // First (lowest) inner cell gid
                           int istride, int ispan,
                           int jstride, int jspan, 
                           int shift) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < ispan * jspan) {
    const int i = tid % ispan;
    const int j = tid / ispan;
    const int inner = offset + i * istride + j * jstride;
    const int outer = inner + shift;
    tnext[outer] = tnext[inner];
  }
}

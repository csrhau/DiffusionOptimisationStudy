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
  // Coalesced read (1D indexing)
  const int block_i = blockIdx.x * blockDim.x;
  const int block_j = blockIdx.y * blockDim.y;
  const int block_k = blockIdx.z * blockDim.z;
  const int i_span = min(blockDim.x + 2, IMAX - block_i);
  const int j_span = min(blockDim.y + 2, JMAX - block_j);
  const int k_span = min(blockDim.z + 2, KMAX - block_k);
  const int j_stride = blockDim.x + 2;
  const int k_stride = (blockDim.x + 2) * (blockDim.y + 2);
  for (int offset = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
       offset < i_span * j_span * k_span;
       offset += blockDim.x * blockDim.y * blockDim.z) {
    const int local_i = offset  % i_span;
    const int local_j = (offset % (i_span * j_span)) / i_span;
    const int local_k = offset / (i_span * j_span);
    const int input_i = block_i + local_i;
    const int input_j = block_j + local_j;
    const int input_k = block_k + local_k;
    const int sid = local_k * k_stride + local_j * j_stride + local_i;
    sdata[sid] = tnow[INDEX3D(input_i, input_j, input_k)];
  }
  __syncthreads();
  // Do compute (using 3-D indexing)
  const int i = block_i + threadIdx.x + 1;
  const int j = block_j + threadIdx.y + 1;
  const int k = block_k + threadIdx.z + 1;
  if (i - 1 < IMAX - 2 && j - 1 < JMAX - 2 && k - 1 < KMAX - 2) {
    const int il = threadIdx.x + 1;
    const int jl = threadIdx.y + 1;
    const int kl = threadIdx.z + 1;
    const int center = kl * k_stride + jl * j_stride + il;
     // Diffuse
    tnext[INDEX3D(i, j, k)] = sdata[center] + cx * (sdata[center-1] - 2.0*sdata[center] + sdata[center+1])
                                            + cy * (sdata[center-j_stride] - 2.0*sdata[center] + sdata[center+j_stride])
                                            + cz * (sdata[center-k_stride]- 2.0*sdata[center] + sdata[center+k_stride]);
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

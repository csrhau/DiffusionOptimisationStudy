#include <iostream> 
#include <vector>
#include <chrono>
#include <numeric>

#include "common.cuh"
#include "simulation.cuh"

int main() {
  const double nu = 0.05;
  const double sigma = 0.25;
  const double width = 2;
  const double height = 2;
  const double dx = width / (IMAX-1);
  const double dy = height / (JMAX-1);
  const double dz = height / (KMAX-1);
  const double dt = sigma * dx * dy * dz / nu;
  const double cx = (nu * dt / (dx * dx));
  const double cy = (nu * dt / (dy * dy));
  const double cz = (nu * dt / (dz * dz));
  // Host Data Initialization
  std::vector<double> thost(IMAX * JMAX * KMAX);
  for (int k = 0; k < KMAX; ++k) {
    for (int j = 0; j < JMAX; ++j) {
      for (int i = 0; i < IMAX; ++i) {
        if (i < HOTCORNER_IMAX && j < HOTCORNER_JMAX && k < HOTCORNER_KMAX) {
          thost[INDEX3D(i, j, k)] = 2.0;
        } else {
          thost[INDEX3D(i, j, k)] = 1.0;
        }
      }
    }
  }
  std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
  // Device Data Initialization
  double *tnow;
  double *tnext;
  cudaMalloc((void **) &tnow, IMAX * JMAX * KMAX * sizeof(double));
  cudaMalloc((void **) &tnext, IMAX * JMAX * KMAX * sizeof(double));
  cudaMemcpy(tnow, thost.data(), IMAX * JMAX * KMAX * sizeof(double), cudaMemcpyHostToDevice);
  // Calculate initial (inner) temperature
  const unsigned long all_cells = (IMAX-2) * (JMAX-2) * (KMAX-2);
  const unsigned long hot_cells = (HOTCORNER_IMAX-1) * (HOTCORNER_JMAX-1) * (HOTCORNER_KMAX-1);
  double expected = hot_cells * 2.0 + (all_cells-hot_cells) * 1.0;
  double temperature = 0.0;
  for (int k = 1; k < KMAX-1; ++k) {
    for (int j = 1; j < JMAX-1; ++j) {
      temperature = std::accumulate(&thost[INDEX3D(1, j, k)], &thost[INDEX3D(IMAX-1, j, k)], temperature);
    }
  }
  std::cout << "Initial Temperature: " << temperature << " Expected: " << expected << std::endl;
  
  const dim3 dim_block(8, 8, 8);
  const dim3 dim_grid((IMAX + dim_block.x - 1) / dim_block.x,
                      (JMAX + dim_block.y - 1) / dim_block.y,
                      (KMAX + dim_block.z - 1) / dim_block.z);

  const dim3 dim_ireflect_block(1, 16, 16);
  const dim3 dim_ireflect_grid(1,
                               (JMAX + dim_block.y - 1) / dim_block.y,
                               (KMAX + dim_block.z - 1) / dim_block.z);

  const dim3 dim_jreflect_block(16, 1, 16);
  const dim3 dim_jreflect_grid((IMAX + dim_block.x - 1) / dim_block.x,
                               1,
                               (KMAX + dim_block.z - 1) / dim_block.z);


  const dim3 dim_kreflect_block(16, 16, 1);
  const dim3 dim_kreflect_grid((IMAX + dim_block.x - 1) / dim_block.x,
                               (JMAX + dim_block.y - 1) / dim_block.y,
                               1);


  const unsigned int smem_bytes = (dim_block.x + 2) * (dim_block.y + 2) * (dim_block.z +2) * sizeof(double);
  std::chrono::steady_clock::time_point t_sim_start = std::chrono::steady_clock::now();
  for (int ts = 0; ts < TIMESTEPS; ++ts) {
    DiffuseKnl<<<dim_grid, dim_block, smem_bytes>>>(tnow, tnext, cx, cy, cz);
    ReflectIKnl<<<dim_ireflect_grid, dim_ireflect_block>>>(tnext);
    ReflectJKnl<<<dim_jreflect_grid, dim_jreflect_block>>>(tnext);
    ReflectKKnl<<<dim_kreflect_grid, dim_kreflect_block>>>(tnext);
    std::swap(tnow, tnext); 
  }
  cudaDeviceSynchronize();
  std::chrono::steady_clock::time_point t_sim_end = std::chrono::steady_clock::now();
  cudaMemcpy(thost.data(), tnow, IMAX * JMAX * KMAX * sizeof(double), cudaMemcpyDeviceToHost);
  temperature = 0.0;
  for (int k = 1; k < KMAX-1; ++k) {
    for (int j = 1; j < JMAX-1; ++j) {
      temperature = std::accumulate(&thost[INDEX3D(1, j, k)], &thost[INDEX3D(IMAX-1, j, k)], temperature);
    }
  }
  cudaFree(tnow);
  cudaFree(tnext);
  cudaDeviceReset();
  std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> runtime = std::chrono::duration_cast<std::chrono::duration<double>>(t_end-t_start);
  std::chrono::duration<double> sim_runtime = std::chrono::duration_cast<std::chrono::duration<double>>(t_sim_end-t_sim_start);
  std::cout << "Final Temperature: " << temperature << " Expected: " << expected << std::endl;
  std::cout << "Time Elapsed (simulation): " << sim_runtime.count() << "s" << std::endl;
  std::cout << "Time Elapsed (total): " << runtime.count() << "s" << std::endl;
  return EXIT_SUCCESS;
}

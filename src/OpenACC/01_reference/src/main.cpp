#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>

#include "parameters.h"

#define INDEX3D(i, j, k) ((((k) * (JMAX) * (IMAX)) + ((j) * (IMAX)) + (i)))

int main() {
  std::cout << "Run Parameters:\n\tIMAX: " << IMAX << "\n\tJMAX: " << JMAX <<
           "\n\tKMAX: " << KMAX << "\n\tTIMESTEPS: " << TIMESTEPS << std::endl;
  const double nu = 0.05;
  const double sigma = 0.25;
  const double width = 2;
  const double height = 2;
  const double dx = width / (IMAX-1);
  const double dy = height / (JMAX-1);
  const double dz = height / (KMAX-1);
  const double dt = sigma * dx * dy * dz / nu;
  const double elems = IMAX * JMAX * KMAX;
  double *restrict tnow = static_cast<double *>(malloc(elems * sizeof(double)));
  double *restrict tnext = static_cast<double *>(malloc(elems * sizeof(double)));;
  for (int k = 0; k < KMAX; ++k) {
    for (int j = 0; j < JMAX; ++j) {
      for (int i = 0; i < IMAX; ++i) {
        if (i < HOTCORNER_IMAX && j < HOTCORNER_JMAX && k < HOTCORNER_KMAX) {
          tnow[INDEX3D(i, j, k)] = 2.0;
        } else {
          tnow[INDEX3D(i, j, k)] = 1.0;
        }
      }
    }
  }
  std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
  // Calculate initial (inner) temperature
  const unsigned long all_cells = (IMAX-2) * (JMAX-2) * (KMAX-2);
  const unsigned long hot_cells = (HOTCORNER_IMAX-1) * (HOTCORNER_JMAX-1) * (HOTCORNER_KMAX-1);
  double expected = hot_cells * 2.0 + (all_cells-hot_cells) * 1.0;
  double temperature = 0.0;
  for (int k = 1; k < KMAX-1; ++k) {
    for (int j = 1; j < JMAX-1; ++j) {
      temperature = std::accumulate(&tnow[INDEX3D(1, j, k)], &tnow[INDEX3D(IMAX-1, j, k)], temperature);
    }
  }
  std::cout << "Initial Temperature: " << temperature << " Expected: " << expected << std::endl;
  std::chrono::steady_clock::time_point t_sim_start = std::chrono::steady_clock::now();
  for (int ts = 0; ts < TIMESTEPS; ++ts) {
    // Diffusion
    #pragma acc parallel loop
    for (int k = 1; k < KMAX-1; ++k) {
      #pragma acc loop
      for (int j = 1; j < JMAX-1; ++j) {
        #pragma acc loop
        for (int i = 1; i < IMAX-1; ++i) {
          tnext[INDEX3D(i, j, k)] = tnow[INDEX3D(i, j, k)] + (nu * dt / (dx * dx)) * (tnow[INDEX3D(i-1, j, k)]-2.0* tnow[INDEX3D(i, j, k)] + tnow[INDEX3D(i+1, j, k)])
                                                           + (nu * dt / (dy * dy)) * (tnow[INDEX3D(i, j-1, k)]-2.0* tnow[INDEX3D(i, j, k)] + tnow[INDEX3D(i, j+1, k)])
                                                           + (nu * dt / (dz * dz)) * (tnow[INDEX3D(i, j, k-1)]-2.0* tnow[INDEX3D(i, j, k)] + tnow[INDEX3D(i, j, k+1)]);
        }
      }
    }
    // Reflective Boundary Condition
    #pragma acc parallel loop
    for (int k = 1; k < KMAX-1; ++k) {
      #pragma acc loop
      for (int j = 1; j < JMAX-1; ++j) {
        tnext[INDEX3D(0, j, k)] = tnext[INDEX3D(1, j, k)];
        tnext[INDEX3D(IMAX-1, j, k)] = tnext[INDEX3D(IMAX-2, j, k)];
      }
      #pragma acc loop
      for (int i = 1; i < IMAX-1; ++i) {
        tnext[INDEX3D(i, 0, k)] = tnext[INDEX3D(i, 1, k)];
        tnext[INDEX3D(i, JMAX-1, k)] = tnext[INDEX3D(i, JMAX-2, k)];
      }
    }
    #pragma acc parallel loop
    for (int j = 1; j < JMAX-1; ++j) {
      #pragma acc loop
      for (int i = 1; i < IMAX-1; ++i) {
        tnext[INDEX3D(i, j, 0)] = tnext[INDEX3D(i, j, 1)];
        tnext[INDEX3D(i, j, KMAX-1)] = tnext[INDEX3D(i, j, KMAX-2)];
      }
    }
    std::swap(tnow, tnext);
  }
  std::chrono::steady_clock::time_point t_sim_end = std::chrono::steady_clock::now();
  temperature = 0.0;
  for (int k = 1; k < KMAX-1; ++k) {
    for (int j = 1; j < JMAX-1; ++j) {
      temperature = std::accumulate(&tnow[INDEX3D(1, j, k)], &tnow[INDEX3D(IMAX-1, j, k)], temperature);
    }
  }
  free(tnow);
  free(tnext);
  std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> runtime = std::chrono::duration_cast<std::chrono::duration<double>>(t_end-t_start);
  std::chrono::duration<double> sim_runtime = std::chrono::duration_cast<std::chrono::duration<double>>(t_sim_end-t_sim_start);
  std::cout << "Final Temperature: " << temperature << " Expected: " << expected << std::endl;
  std::cout << "Time Elapsed (simulation): " << sim_runtime.count() << "s" << std::endl;
  std::cout << "Time Elapsed (total): " << runtime.count() << "s" << std::endl;
  return EXIT_SUCCESS;
}

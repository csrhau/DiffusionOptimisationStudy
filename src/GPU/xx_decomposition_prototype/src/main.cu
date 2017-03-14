#include "main.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "common.h"
#include "simulation.h"
#include "time_functions.cuh"

int main(void) {
  // Decomposition values
    // Simulation values
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
  // Initialize state
  double *host_state[2];
  // double *device_state[2];
  host_state[0] = (double *) malloc(IMAX * JMAX * KMAX * sizeof(double));
  host_state[1] = (double *) malloc(IMAX * JMAX * KMAX * sizeof(double));
  // Host Data Initialization
  for (int k = 0; k < KMAX; ++k) {
    for (int j = 0; j < JMAX; ++j) {
      for (int i = 0; i < IMAX; ++i) {
        size_t center = k * JMAX * IMAX + j * IMAX + i;
        if (i < HOTCORNER_IMAX && j < HOTCORNER_JMAX && k < HOTCORNER_KMAX) {
          host_state[0][center] = 2.0;
        } else {
          host_state[0][center] = 1.0;
        }
      }
    }
  }
  struct timespec start, simstart, simend, end;
  RecordTime(&start);
  // Device Data Initialization
  // cudaMalloc((void **) &device_state[0], IMAX * JMAX * KMAX * sizeof(double));
  // cudaMalloc((void **) &device_state[1], IMAX * JMAX * KMAX * sizeof(double));
  // cudaMemcpy(device_state[0], host_state[0], IMAX * JMAX * KMAX * sizeof(double), cudaMemcpyHostToDevice);
  // Calculate initial (inner) temperature
  const unsigned long all_cells = (IMAX-2) * (JMAX-2) * (KMAX-2);
  const unsigned long hot_cells = (HOTCORNER_IMAX-1) * (HOTCORNER_JMAX-1) * (HOTCORNER_KMAX-1);
  double expected = hot_cells * 2.0 + (all_cells-hot_cells) * 1.0;
  double temperature = 0.0;
  for (int k = 1; k < KMAX-1; ++k) {
    for (int j = 1; j < JMAX-1; ++j) {
      for (int i = 1; i < IMAX-1; ++i) {
        size_t center = k * JMAX * IMAX + j * IMAX + i;
        temperature += host_state[0][center];
      }
    }
  }
  printf("Initial Temperature: %f Expected: %f\n", temperature, expected);
  RecordTime(&simstart);
  RecursiveTrapezoid(host_state,
                     cx, cy, cz,
                     0, TIMESTEPS,
                     1, 0, IMAX-1, 0, IMAX,
                     1, 0, JMAX-1, 0, JMAX,
                     1, 0, KMAX-1, 0, KMAX);
  RecordTime(&simend);
  temperature = 0.0;
  for (int k = 1; k < KMAX-1; ++k) {
    for (int j = 1; j < JMAX-1; ++j) {
      for (int i = 1; i < IMAX-1; ++i) {
        size_t center = k * JMAX * IMAX + j * IMAX + i;
        temperature += host_state[TIMESTEPS & 1][center];
      }
    }
  }

  for (int k = 0; k < KMAX-0; ++k) {
    for (int j = 0; j < JMAX-0; ++j) {
      for (int i = 0; i < IMAX-0; ++i) {
        size_t center = k * JMAX * IMAX + j * IMAX + i;
        std::cout << host_state[TIMESTEPS & 1][center] << " ";
      }
      std::cout << std::endl;
    }
      std::cout << std::endl;
      std::cout << std::endl;
  }



  printf("Final Temperature: %f Expected: %f\n", temperature, expected);
  RecordTime(&end);
  printf("Time Elapsed (simulation): %fs\n", TimeDifference(&simstart, &simend));
  printf("Time Elapsed (total): %fs\n", TimeDifference(&start, &end));
  return EXIT_SUCCESS;
}

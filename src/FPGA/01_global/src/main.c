#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

#include "xcl_world.h"
#include "xcl_buffer.h"
#include "simulation.h"
#include "common.h"

int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "%s: A FPGA-Based Explicit Solver for the Diffusion Equation\n"
                    "Usage: %s fpga-device fpga-binary\n", argv[0], argv[0]); 
    return EXIT_FAILURE;
  }

  struct XCLWorld world;
  XCLSetup(VENDOR_STRING, argv[1], argv[2], &world);
  if (world.status != CL_SUCCESS) {
    fprintf(stderr, "Failed to initialise OpenCL. Error Code: %d\n", world.status);
    return EXIT_FAILURE;
  } else {
    printf("OpenCL Environment Setup Complete.\n\tVendor: %s\n\tDevice: %s\n\tBinary: %s\n", 
        world.vendor_name, world.device_name, world.binary_name);
  }
  const float nu = 0.05;
  const float sigma = 0.15;
  const float width = 2;
  const float height = 2;
  const float dx = width / (IMAX-1);
  const float dy = height / (JMAX-1);
  const float dz = height / (KMAX-1);
  const float dt = sigma * dx * dy * dz / nu;
  const float cx = (nu * dt / (dx * dx));
  const float cy = (nu * dt / (dy * dy));
  const float cz = (nu * dt / (dz * dz));
  const unsigned long all_cells = (IMAX-2) * (JMAX-2) * (KMAX-2);
  const unsigned long hot_cells = (HOTCORNER_IMAX-1) * (HOTCORNER_JMAX-1) * (HOTCORNER_KMAX-1);
  float expected = hot_cells * 2.0 + (all_cells-hot_cells) * 1.0;
  struct Simulation simulation;
  SimulationSetup(IMAX, JMAX, KMAX, cx, cy, cz, &world, &simulation);
  if (world.status != CL_SUCCESS) {
    fprintf(stderr, "Failed to set up simulation: %d\n", world.status);
    return EXIT_FAILURE;
  } else {
    printf("Simulation Setup Complete.\n\tIMAX: %d\tHOTCORNER_IMAX: %d\n\t"
           "JMAX: %d\tHOTCORNER_JMAX: %d\n\tKMAX: %d\tHOTCORNER_KMAX: %d\n\t"
           "Expected Temperature: %f\n", IMAX, HOTCORNER_IMAX, JMAX, HOTCORNER_JMAX, KMAX, HOTCORNER_KMAX, expected);
  }
  // Host Data Initialization
  for (int k = 0; k < KMAX; ++k) {
    for (int j = 0; j < JMAX; ++j) {
      for (int i = 0; i < IMAX; ++i) {
        size_t center = k * JMAX * IMAX + j * IMAX + i;
        if (i < HOTCORNER_IMAX && j < HOTCORNER_JMAX && k < HOTCORNER_KMAX) {
          simulation.field_a.host_data[center] = 2.0;
        } else {
          simulation.field_a.host_data[center] = 1.0;
        }
      }
    }
  }
  // Push input data to device
  printf("Sending data to FPGA\n");
  SimulationPushData(&world, &simulation);
  // Calculate Initial Temperature
  printf("Computing initial temperature on FPGA\n");
  SimulationComputeTemperature(&world, &simulation);
  printf("Pulling back temperature from FPGA\n");
  SimulationPullRegisters(&world, &simulation);
  float measured = simulation.registers.host_data[TEMPERATURE];
  printf("Initial Temperature: %f (expected), %f (measured), %f (error)\n",expected, measured, measured-expected);
  // Run Simulation
  printf("Running simulation...\n");
  for (unsigned ts = 1; ts <= TIMESTEPS; ++ts) {
    SimulationDiffuse(&world, &simulation);
    if (ts % 10 == 0) {
      printf("Timestep %d\n", ts);
    }
  }
  printf("Simulation complete at timestep %d\n", TIMESTEPS);
  // Calculate Final Temperature
  printf("Computing final temperature on FPGA\n");
  SimulationComputeTemperature(&world, &simulation);
  printf("Retrieving registers from FPGA\n");
  SimulationPullRegisters(&world, &simulation);
  measured = simulation.registers.host_data[TEMPERATURE];
  printf("Final Temperature: %f (expected), %f (measured), %f (error)\n",expected, measured, measured-expected);
  // Dump to screen
  /*
  for (int k = 0; k < KMAX; ++k) {
    for (int j = 0; j < JMAX; ++j) {
      for (int i = 0; i < IMAX; ++i) {
        size_t center = k * JMAX * IMAX + j * IMAX + i;
        printf("%f ", simulation.field_b.host_data[center]);
      }
      printf("\n");
    }
      printf("\n");
      printf("\n");
  }
  */
  // Cleanup
  SimulationTeardown(&simulation);
  XCLTeardown(&world);
  return EXIT_SUCCESS;
}

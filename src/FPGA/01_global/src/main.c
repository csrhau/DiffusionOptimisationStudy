#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

#include "xcl_world.h"
#include "xcl_buffer.h"
#include "simulation.h"
#include "common.h"

int main(void) {
  struct XCLWorld world;
  XCLSetup(VENDOR_STRING, DEVICE_STRING, BINARY_STRING, &world);
  if (world.status != CL_SUCCESS) {
    fprintf(stderr, "Failed to initialise OpenCL. Error Code: %d\n", world.status);
    return EXIT_FAILURE;
  } else {
    printf("OpenCL Environment Setup Complete.\n\tVendor: %s\n\tDevice: %s\n\tBinary: %s\n", 
        world.vendor_name, world.device_name, world.binary_name);
  }
  struct Simulation simulation;
  SimulationSetup(IMAX, JMAX, KMAX, &world, &simulation);
  if (world.status != CL_SUCCESS) {
    fprintf(stderr, "Failed to set up simulation: %d\n", world.status);
    return EXIT_FAILURE;
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
  const unsigned long all_cells = (IMAX-2) * (JMAX-2) * (KMAX-2);
  const unsigned long hot_cells = (HOTCORNER_IMAX-1) * (HOTCORNER_JMAX-1) * (HOTCORNER_KMAX-1);
  float expected = hot_cells * 2.0 + (all_cells-hot_cells) * 1.0;
  // Push input data to device
  SimulationPushData(&world, &simulation);
  // Calculate Initial Temperature
  SimulationComputeTemperature(&world, &simulation);
  SimulationPullData(&world, &simulation);
  float measured = simulation.registers.host_data[TEMPERATURE];
  printf("Initial Temperature: %f (expected), %f (measured), %f (error)\n",expected, measured, measured-expected);
  // Run Simulation
  SimulationDiffuse(&world, &simulation);
  // Calculate Final Temperature
  SimulationComputeTemperature(&world, &simulation);
  SimulationPullData(&world, &simulation);
  measured = simulation.registers.host_data[TEMPERATURE];
  printf("Final Temperature: %f (expected), %f (measured), %f (error)\n",expected, measured, measured-expected);
  // Cleanup
  SimulationTeardown(&simulation);
  XCLTeardown(&world);
  return EXIT_SUCCESS;
}

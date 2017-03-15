#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

#include "xcl_world.h"
#include "xcl_buffer.h"
#include "simulation.h"

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
  SimulationSetup(32, &world, &simulation);
  if (world.status != CL_SUCCESS) {
    fprintf(stderr, "Failed to set up simulation: %d\n", world.status);
    return EXIT_FAILURE;
  }
  // Set input values
  for (int i = 0; i < 32; ++i) {
    simulation.a.host_data[i] = i;
    simulation.b.host_data[i] = i;
  }
  // Push input data to device
  SimulationPushData(&world, &simulation);
  // Run simulation
  SimulationStep(&world, &simulation);
  // Pull output data back from device
  SimulationPullData(&world, &simulation);
  if (world.status != CL_SUCCESS) {
    fprintf(stderr, "Failed to pull buffer c: %d\n", world.status);
    return EXIT_FAILURE;
  }
  // Validate answer
  for (int i = 0; i < 32; ++i) {
    printf("%d:\t%f + %f = %f\n", i, simulation.a.host_data[i], simulation.b.host_data[i], simulation.c.host_data[i]);
  }
  // Cleanup
  SimulationTeardown(&simulation);
  XCLTeardown(&world);
  return EXIT_SUCCESS;
}

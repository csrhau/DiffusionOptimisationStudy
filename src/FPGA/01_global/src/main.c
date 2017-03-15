#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

#include "xcl_world.h"
#include "xcl_buffers.h"
#include "simulation.h"

int main(void) {
  struct XCLWorld xcl_world;
  XCLSetup(VENDOR_STRING, DEVICE_STRING, BINARY_STRING, &xcl_world);
  if (xcl_world.status != CL_SUCCESS) {
    fprintf(stderr, "Failed to initialise OpenCL. Error Code: %d\n", xcl_world.status);
    return EXIT_FAILURE;
  } else {
    printf("OpenCL Environment Setup Complete.\n\tVendor: %s\n\tDevice: %s\n\tBinary: %s\n", 
        xcl_world.vendor_name, xcl_world.device_name, xcl_world.binary_name);
  }
  struct Simulation simulation;
  SimulationSetup(1024, &xcl_world, &simulation);
  if (xcl_world.status != CL_SUCCESS) {
    fprintf(stderr, "Failed to set up simulation: %d\n", xcl_world.status);
    return EXIT_FAILURE;
  }
  // Set input values
  for (int i = 0; i < 1024; ++i) {
    simulation.a.host_data[i] = i;
    simulation.b.host_data[i] = i;
  }
  // Push input data to device
  XCLPushDoubleBuffer(&xcl_world, &simulation.a);
  if (xcl_world.status != CL_SUCCESS) {
    fprintf(stderr, "Failed to push buffer a: %d\n", xcl_world.status);
    return EXIT_FAILURE;
  }
  XCLPushDoubleBuffer(&xcl_world, &simulation.b);
  if (xcl_world.status != CL_SUCCESS) {
    fprintf(stderr, "Failed to push buffer b: %d\n", xcl_world.status);
    return EXIT_FAILURE;
  }
  // Pull output data from device
  XCLPullDoubleBuffer(&xcl_world, &simulation.c);
  if (xcl_world.status != CL_SUCCESS) {
    fprintf(stderr, "Failed to pull buffer c: %d\n", xcl_world.status);
    return EXIT_FAILURE;
  }
  // Validate answer
  SimulationTeardown(&simulation);
  XCLTeardown(&xcl_world);
  return EXIT_SUCCESS;
}

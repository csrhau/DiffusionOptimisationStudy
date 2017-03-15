#include <stdio.h>
#include <stdlib.h>
#include "xcl_tools.h"

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
  XCLTeardown(&xcl_world);
  return EXIT_SUCCESS;
}

#include "xcl_kernel.h"

#include <string.h>
#include <CL/cl.h>

#include "xcl_world.h"

void XCLKernelSetup(char *name,
                    struct XCLWorld* world,
                    struct XCLKernel* kernel) {
  kernel->name = strdup(name);
  kernel->kernel = clCreateKernel(world->program, name, &world->status);
}

void XCLKernelTeardown(struct XCLKernel* kernel) {
  if (kernel) {
    if (kernel->name) {
      free(kernel->name);
      kernel->name = NULL;
    }
    if (kernel->kernel) {
      clReleaseKernel(kernel->kernel);
      kernel->kernel = 0;
    }
  }
}


void XCLKernelSetArg(unsigned argnum,
                     size_t size,
                     const void *value,
                     struct XCLWorld* world,
                     struct XCLKernel* kernel) {
  world->status = clSetKernelArg(kernel->kernel, argnum, size, value);
}


cl_int XCLKernelInvoke(struct XCLWorld* world, struct XCLKernel* kernel) {
  size_t global_work_size = 1;
  size_t local_work_size = 1;
  return clEnqueueNDRangeKernel(world->queue, kernel->kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
}

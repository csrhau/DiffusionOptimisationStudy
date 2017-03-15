#pragma once

#include <CL/cl.h>

struct XCLWorld;

struct XCLKernel {
  char *name;
  cl_kernel kernel;
};

void XCLKernelSetup(char *name,
                    struct XCLWorld* world,
                    struct XCLKernel* kernel);

void XCLKernelTeardown(struct XCLKernel* kernel);

void XCLKernelSetArg(unsigned argnum,
                     size_t size,
                     const void *value,
                     struct XCLWorld* world,
                     struct XCLKernel* kernel);
cl_int XCLKernelInvoke(struct XCLWorld* world, struct XCLKernel* kernel);

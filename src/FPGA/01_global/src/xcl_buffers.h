#pragma once

#include <CL/cl.h>

struct XCLWorld;

struct XCLDoubleBuffer {
  size_t elements;
  size_t bytes;
  double *host_data;
  cl_mem fpga_data;
  cl_mem_flags flags;
};

void XCLDoubleBufferSetup(size_t elements,
                          cl_mem_flags flags,
                          struct XCLWorld* world,
                          struct XCLDoubleBuffer* buffer);

void XCLDoubleBufferTeardown(struct XCLDoubleBuffer* buffer);

void XCLPushDoubleBuffer(struct XCLWorld* world, struct XCLDoubleBuffer *buffer);
void XCLPullDoubleBuffer(struct XCLWorld* world, struct XCLDoubleBuffer *buffer);

#pragma once

#include <CL/cl.h>

struct XCLWorld;

struct XCLFloatBuffer {
  unsigned elements;
  size_t bytes;
  float *host_data;
  cl_mem fpga_data;
  cl_mem_flags flags;
};

void XCLFloatBufferSetup(unsigned elements,
                          cl_mem_flags flags,
                          struct XCLWorld* world,
                          struct XCLFloatBuffer* buffer);

void XCLFloatBufferTeardown(struct XCLFloatBuffer* buffer);

void XCLPushFloatBuffer(struct XCLWorld* world, struct XCLFloatBuffer *buffer);
void XCLPullFloatBuffer(struct XCLWorld* world, struct XCLFloatBuffer *buffer);

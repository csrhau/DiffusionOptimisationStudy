#include "xcl_buffers.h"

#include <stdlib.h>

#include "xcl_world.h"

void XCLDoubleBufferSetup(size_t elements,
                          cl_mem_flags flags,
                          struct XCLWorld* world,
                          struct XCLDoubleBuffer* buffer) {
  buffer->elements = elements;
  buffer->bytes = elements * sizeof(double);
  buffer->flags = flags;
  buffer->host_data = (double *) malloc(buffer->bytes);
  buffer->fpga_data = clCreateBuffer(world->context, flags, buffer->bytes, NULL, &world->status);
}

void XCLDoubleBufferTeardown(struct XCLDoubleBuffer* buffer) {
  if (buffer->host_data) {
    free(buffer->host_data);
    buffer->host_data = NULL;
  }
  if (buffer->fpga_data) {
    clReleaseMemObject(buffer->fpga_data);
    buffer->fpga_data = 0;
  }
}

void XCLPushDoubleBuffer(struct XCLWorld* world, struct XCLDoubleBuffer *buffer) {
  world->status = clEnqueueWriteBuffer(world->queue, buffer->fpga_data, CL_TRUE,
      0, buffer->bytes, buffer->host_data, 0, NULL, NULL);
}

void XCLPullDoubleBuffer(struct XCLWorld* world, struct XCLDoubleBuffer *buffer) {
  world->status = clEnqueueReadBuffer(world->queue, buffer->fpga_data, CL_TRUE,
      0, buffer->bytes, buffer->host_data, 0, NULL, NULL);
}

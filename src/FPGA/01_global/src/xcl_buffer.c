#include "xcl_buffer.h"

#include <stdlib.h>

#include "xcl_world.h"

void XCLFloatBufferSetup(size_t elements,
                          cl_mem_flags flags,
                          struct XCLWorld* world,
                          struct XCLFloatBuffer* buffer) {
  buffer->elements = elements;
  buffer->bytes = elements * sizeof(float);
  buffer->flags = flags;
  buffer->host_data = (float *) malloc(buffer->bytes);
  buffer->fpga_data = clCreateBuffer(world->context, flags, buffer->bytes, NULL, &world->status);
}

void XCLFloatBufferTeardown(struct XCLFloatBuffer* buffer) {
  if (buffer) {
    if (buffer->host_data) {
      free(buffer->host_data);
      buffer->host_data = NULL;
    }
    if (buffer->fpga_data) {
      clReleaseMemObject(buffer->fpga_data);
      buffer->fpga_data = 0;
    }
  }
}

void XCLPushFloatBuffer(struct XCLWorld* world, struct XCLFloatBuffer *buffer) {
  world->status = clEnqueueWriteBuffer(world->queue, buffer->fpga_data, CL_TRUE,
      0, buffer->bytes, buffer->host_data, 0, NULL, NULL);
}

void XCLPullFloatBuffer(struct XCLWorld* world, struct XCLFloatBuffer *buffer) {
  world->status = clEnqueueReadBuffer(world->queue, buffer->fpga_data, CL_TRUE,
      0, buffer->bytes, buffer->host_data, 0, NULL, NULL);
}

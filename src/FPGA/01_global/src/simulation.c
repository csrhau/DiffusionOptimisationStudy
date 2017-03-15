#include "simulation.h"

#include <stdlib.h>

#include "xcl_world.h"
#include "xcl_buffer.h"

void SimulationSetup(size_t elements,
                     struct XCLWorld* world,
                     struct Simulation* simulation) {
  XCLFloatBufferSetup(elements, CL_MEM_READ_ONLY, world, &simulation->a);
  if (world->status != CL_SUCCESS) { return; }
  XCLFloatBufferSetup(elements, CL_MEM_READ_ONLY, world, &simulation->b);
  if (world->status != CL_SUCCESS) { return; }
  XCLFloatBufferSetup(elements, CL_MEM_WRITE_ONLY, world, &simulation->c);
  if (world->status != CL_SUCCESS) { return; }
  XCLKernelSetup("krnl_vadd", world, &simulation->kernel);
  if (world->status != CL_SUCCESS) { return; }
  XCLKernelSetArg(0, sizeof(cl_mem), &simulation->a.fpga_data, world, &simulation->kernel);
  if (world->status != CL_SUCCESS) { return; }
  XCLKernelSetArg(1, sizeof(cl_mem), &simulation->b.fpga_data, world, &simulation->kernel);
  if (world->status != CL_SUCCESS) { return; }
  XCLKernelSetArg(2, sizeof(cl_mem), &simulation->c.fpga_data, world, &simulation->kernel);
}

void SimulationTeardown(struct Simulation *simulation) {
  if (simulation) {
    XCLKernelTeardown(&simulation->kernel);
    XCLFloatBufferTeardown(&simulation->a);
    XCLFloatBufferTeardown(&simulation->b);
    XCLFloatBufferTeardown(&simulation->c);
  }
}

void SimulationPushData(struct XCLWorld *world,
                        struct Simulation *simulation) {
  // Push input data to device
  XCLPushFloatBuffer(world, &simulation->a);
  if (world->status != CL_SUCCESS) { return; }
  XCLPushFloatBuffer(world, &simulation->b);
}

void SimulationPullData(struct XCLWorld *world,
                        struct Simulation *simulation) {
  XCLPullFloatBuffer(world, &simulation->c);
}

void SimulationStep(struct XCLWorld *world,
                    struct Simulation *simulation) {
  XCLKernelInvoke(world, &simulation->kernel);
}

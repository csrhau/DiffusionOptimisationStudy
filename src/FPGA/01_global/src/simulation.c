#include "simulation.h"

#include <stdlib.h>

#include "xcl_world.h"
#include "xcl_buffer.h"
#include "common.h"

void SimulationSetup(size_t imax,
                     size_t jmax,
                     size_t kmax,
                     float cx,
                     float cy,
                     float cz,
                     struct XCLWorld* world,
                     struct Simulation* simulation) {
  simulation->ts = 0;
  simulation->imax = imax;
  simulation->jmax = jmax;
  simulation->kmax = kmax;
  simulation->cx = cx;
  simulation->cy = cy;
  simulation->cz = cz;
  XCLFloatBufferSetup(imax*jmax*kmax, CL_MEM_READ_ONLY, world, &simulation->field_a);
  if (world->status != CL_SUCCESS) { return; }
  XCLFloatBufferSetup(imax*jmax*kmax, CL_MEM_READ_ONLY, world, &simulation->field_b);
  if (world->status != CL_SUCCESS) { return; }
  XCLFloatBufferSetup(1, CL_MEM_WRITE_ONLY, world, &simulation->registers);
  if (world->status != CL_SUCCESS) { return; }
  XCLKernelSetup("krnl_temperature", world, &simulation->temperature_kernel);
  if (world->status != CL_SUCCESS) { return; }
  XCLKernelSetup("krnl_diffuse", world, &simulation->diffusion_kernel);
  if (world->status != CL_SUCCESS) { return; }

  XCLKernelSetArg(0, sizeof(cl_mem), &simulation->field_a.fpga_data, world, &simulation->temperature_kernel);
  if (world->status != CL_SUCCESS) { return; }
  XCLKernelSetArg(1, sizeof(cl_mem), &simulation->registers.fpga_data, world, &simulation->temperature_kernel);
  if (world->status != CL_SUCCESS) { return; }
}

void SimulationTeardown(struct Simulation *simulation) {
  if (simulation) {
    simulation->imax = 0;
    simulation->jmax = 0;
    simulation->kmax = 0;
    XCLKernelTeardown(&simulation->temperature_kernel);
    XCLFloatBufferTeardown(&simulation->field_a);
    XCLFloatBufferTeardown(&simulation->field_b);
    XCLFloatBufferTeardown(&simulation->registers);
  }
}

void SimulationPushData(struct XCLWorld *world,
                        struct Simulation *simulation) {
  // Push input data to device
  XCLPushFloatBuffer(world, &simulation->field_a);
  if (world->status != CL_SUCCESS) { return; }
  XCLPushFloatBuffer(world, &simulation->field_b);
}

void SimulationPullData(struct XCLWorld *world,
                        struct Simulation *simulation) {
  XCLPullFloatBuffer(world, &simulation->field_a);
  if (world->status != CL_SUCCESS) { return; }
  XCLPullFloatBuffer(world, &simulation->field_b);
}

void SimulationPushRegisters(struct XCLWorld *world,
                             struct Simulation *simulation) {
  XCLPushFloatBuffer(world, &simulation->registers);
}

void SimulationPullRegisters(struct XCLWorld *world,
                             struct Simulation *simulation) {
  XCLPullFloatBuffer(world, &simulation->registers);
}

void SimulationDiffuse(struct XCLWorld *world,
                       struct Simulation *simulation) {
  struct XCLFloatBuffer *from_field;
  struct XCLFloatBuffer *to_field;
  if (simulation->ts & 1) {
    from_field = &simulation->field_b;
    to_field = &simulation->field_a;
  } else {
    from_field = &simulation->field_a;
    to_field = &simulation->field_b;
  }
  XCLKernelSetArg(0, sizeof(cl_mem), &from_field->fpga_data, world, &simulation->diffusion_kernel);
  if (world->status != CL_SUCCESS) { return; }
  XCLKernelSetArg(1, sizeof(cl_mem), &to_field->fpga_data, world, &simulation->diffusion_kernel);
  if (world->status != CL_SUCCESS) { return; }
  XCLKernelSetArg(2, sizeof(float), &simulation->cx, world, &simulation->diffusion_kernel);
  if (world->status != CL_SUCCESS) { return; }
  XCLKernelSetArg(3, sizeof(float), &simulation->cy, world, &simulation->diffusion_kernel);
  if (world->status != CL_SUCCESS) { return; }
  XCLKernelSetArg(4, sizeof(float), &simulation->cz, world, &simulation->diffusion_kernel);
  if (world->status != CL_SUCCESS) { return; }
  XCLKernelInvoke(world, &simulation->diffusion_kernel);
  simulation->ts++;
}


void SimulationComputeTemperature(struct XCLWorld *world,
                                  struct Simulation *simulation) {
  struct XCLFloatBuffer *field;
  if (simulation->ts & 1) {
    field = &simulation->field_b;
  } else {
    field = &simulation->field_a;
  }
  XCLKernelSetArg(0, sizeof(cl_mem), &field->fpga_data, world, &simulation->temperature_kernel);
  if (world->status != CL_SUCCESS) { return; }
  XCLKernelSetArg(1, sizeof(cl_mem), &simulation->registers.fpga_data, world, &simulation->temperature_kernel);
  if (world->status != CL_SUCCESS) { return; }
  XCLKernelInvoke(world, &simulation->temperature_kernel);
}

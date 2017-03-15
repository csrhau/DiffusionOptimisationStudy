#include "simulation.h"

#include <stdlib.h>

#include "xcl_world.h"
#include "xcl_buffers.h"

void SimulationSetup(size_t elements,
                     struct XCLWorld* world,
                     struct Simulation* simulation) {
  simulation->world = world;
  XCLDoubleBufferSetup(elements, CL_MEM_READ_ONLY, world, &simulation->a);
  if (world->status != CL_SUCCESS) { return; }
  XCLDoubleBufferSetup(elements, CL_MEM_READ_ONLY, world, &simulation->b);
  if (world->status != CL_SUCCESS) { return; }
  XCLDoubleBufferSetup(elements, CL_MEM_WRITE_ONLY, world, &simulation->c);
  if (world->status != CL_SUCCESS) { return; }
}

void SimulationTeardown(struct Simulation *simulation) {
  XCLDoubleBufferTeardown(&simulation->a);
  XCLDoubleBufferTeardown(&simulation->b);
  XCLDoubleBufferTeardown(&simulation->c);
}

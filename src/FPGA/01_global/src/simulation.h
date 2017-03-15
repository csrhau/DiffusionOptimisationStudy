#pragma once

#include "xcl_buffer.h"
#include "xcl_kernel.h"

#include <CL/cl.h>

struct XCLWorld; 

struct Simulation {
  struct XCLKernel kernel;
  struct XCLFloatBuffer a;
  struct XCLFloatBuffer b;
  struct XCLFloatBuffer c;
};

void SimulationSetup(size_t elements,
                     struct XCLWorld* world,
                     struct Simulation* simulation);
void SimulationTeardown(struct Simulation *simulation);

void SimulationPushData(struct XCLWorld* world, struct Simulation *simulation);
void SimulationPullData(struct XCLWorld* world, struct Simulation *simulation);

void SimulationStep(struct XCLWorld* world, struct Simulation* simulation);

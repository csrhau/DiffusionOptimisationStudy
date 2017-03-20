#pragma once

#include "xcl_buffer.h"
#include "xcl_kernel.h"

#include <CL/cl.h>

struct XCLWorld; 

struct Simulation {
  unsigned long ts;
  size_t imax; 
  size_t jmax; 
  size_t kmax; 
  float cx;
  float cy;
  float cz;
  struct XCLKernel temperature_kernel;
  struct XCLKernel diffusion_kernel;
  struct XCLFloatBuffer field_a;
  struct XCLFloatBuffer field_b;
  struct XCLFloatBuffer registers;
};

void SimulationSetup(size_t imax,
                     size_t jmax,
                     size_t kmax,
                     float cx,
                     float cy,
                     float cz,
                     struct XCLWorld* world,
                     struct Simulation* simulation);
void SimulationTeardown(struct Simulation *simulation);

void SimulationPushData(struct XCLWorld* world, struct Simulation *simulation);
void SimulationPullData(struct XCLWorld* world, struct Simulation *simulation);

void SimulationPushRegisters(struct XCLWorld* world, struct Simulation *simulation);
void SimulationPullRegisters(struct XCLWorld* world, struct Simulation *simulation);




void SimulationDiffuse(struct XCLWorld* world, struct Simulation* simulation);
void SimulationComputeTemperature(struct XCLWorld* world, struct Simulation* simulation);

#pragma once

#include "xcl_buffers.h"

struct XCLWorld; 

struct Simulation {
  struct XCLWorld* world;
  struct XCLDoubleBuffer a;
  struct XCLDoubleBuffer b;
  struct XCLDoubleBuffer c;
};

void SimulationSetup(size_t elements,
                     struct XCLWorld* world,
                     struct Simulation* simulation);

void SimulationTeardown(struct Simulation *simulation);

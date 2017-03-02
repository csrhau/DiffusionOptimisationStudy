#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "common.h"

#include "simulation.h"

TEST(SimulationTests, OneEqualsOne) {
  EXPECT_EQ(1, 1);
}

TEST(SimulationTests, ConservationOfEnergy) {
  const double nu = 0.05;
  const double sigma = 0.25;
  const double width = 2;
  const double height = 2;
  const double dx = width / (IMAX-1);
  const double dy = height / (JMAX-1);
  const double dz = height / (KMAX-1);
  const double dt = sigma * dx * dy * dz / nu;
  const double cx = (nu * dt / (dx * dx));
  const double cy = (nu * dt / (dy * dy));
  const double cz = (nu * dt / (dz * dz));
  // Initialize state
  double *state[2];
  state[0] = (double *) malloc(IMAX * JMAX * KMAX * sizeof(double));
  state[1] = (double *) malloc(IMAX * JMAX * KMAX * sizeof(double));
  // Host Data Initialization
  for (int k = 0; k < KMAX; ++k) {
    for (int j = 0; j < JMAX; ++j) {
      for (int i = 0; i < IMAX; ++i) {
        size_t center = k * JMAX * IMAX + j * IMAX + i;
        if (i < HOTCORNER_IMAX && j < HOTCORNER_JMAX && k < HOTCORNER_KMAX) {
          state[0][center] = 2.0;
        } else {
          state[0][center] = 1.0;
        }
      }
    }
  }
  const unsigned long all_cells = (IMAX-2) * (JMAX-2) * (KMAX-2);
  const unsigned long hot_cells = (HOTCORNER_IMAX-1) * (HOTCORNER_JMAX-1) * (HOTCORNER_KMAX-1);
  double expected = hot_cells * 2.0 + (all_cells-hot_cells) * 1.0;
  RecursiveTrapezoid(state,
                    cx, cy, cz,
                    0, TIMESTEPS,
                    1, 0, IMAX-1, 0, IMAX,
                    1, 0, JMAX-1, 0, JMAX,
                    1, 0, KMAX-1, 0, KMAX);
  double temperature = 0.0;
  for (int k = 1; k < KMAX-1; ++k) {
    for (int j = 1; j < JMAX-1; ++j) {
      for (int i = 1; i < IMAX-1; ++i) {
        size_t center = k * JMAX * IMAX + j * IMAX + i;
        temperature += state[TIMESTEPS & 1][center];
      }
    }
  }
  EXPECT_NEAR(expected, temperature, 1.0E-7);
}

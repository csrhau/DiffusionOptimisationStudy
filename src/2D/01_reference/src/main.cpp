#include <iostream> 
#include <vector>
#include <chrono>
#include <numeric>

#define TIMESTEPS 100

#define STENCIL_RADIUS 1

#define IMAX 100
#define JMAX 100

#define HOTCORNER_IMAX 25
#define HOTCORNER_JMAX 25

#define INDEX2D(i, j) ((((j) * (IMAX)) + (i)))

int main() {
  const double nu = 0.05;
  const double sigma = 0.25;
  const double width = 2;
  const double height = 2;
  const double dx = width / (IMAX-1);
  const double dy = height / (JMAX-1);
  const double dt = sigma * dx * dy / nu;
  std::vector<double> tnow(IMAX * JMAX);
  std::vector<double> tnext(IMAX * JMAX);
  for (int j = 0; j < JMAX; ++j) {
    for (int i = 0; i < IMAX; ++i) {
      if (i < HOTCORNER_IMAX && j < HOTCORNER_JMAX) {
        tnow[INDEX2D(i, j)] = 2.0;
      } else {
        tnow[INDEX2D(i, j)] = 1.0;
      }
    }
  }
  std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
  // Calculate initial (inner) temperature
  const unsigned long all_cells = (IMAX-2) * (JMAX-2);
  const unsigned long hot_cells = (HOTCORNER_IMAX-1) * (HOTCORNER_JMAX-1);
  double expected = hot_cells * 2.0 + (all_cells-hot_cells) * 1.0;
  double temperature = 0.0;
  for (int j = 1; j < JMAX-1; ++j) {
    temperature = std::accumulate(&tnow[INDEX2D(1, j)], &tnow[INDEX2D(IMAX-1, j)], temperature);
  }
  std::cout << "Initial Temperature: " << temperature << " Expected: " << expected << std::endl;
  for (int ts = 0; ts < TIMESTEPS; ++ts) {
    // Diffusion
    for (int j = 1; j < JMAX-1; ++j) {
      for (int i = 1; i < IMAX-1; ++i) {
        tnext[INDEX2D(i, j)] = tnow[INDEX2D(i, j)] + (nu * dt / (dx * dx)) * (tnow[INDEX2D(i-1, j)] - 2.0 * tnow[INDEX2D(i, j)] + tnow[INDEX2D(i+1, j)])
                                                   + (nu * dt / (dy * dy)) * (tnow[INDEX2D(i, j-1)] - 2.0 * tnow[INDEX2D(i, j)] + tnow[INDEX2D(i, j+1)]);
      }
    }
    // Reflective Boundary Condition
    for (int j = 1; j < JMAX-1; ++j) {
      tnext[INDEX2D(0, j)] = tnext[INDEX2D(1, j)];
      tnext[INDEX2D(IMAX-1, j)] = tnext[INDEX2D(IMAX-2, j)];
    }
    for (int i = 1; i < IMAX-1; ++i) {
      tnext[INDEX2D(i, 0)] = tnext[INDEX2D(i, 1)];
      tnext[INDEX2D(i, JMAX-1)] = tnext[INDEX2D(i, JMAX-2)];
    }
    std::swap(tnow, tnext); 
  }
  temperature = 0.0;
  for (int j = 1; j < JMAX-1; ++j) {
    temperature = std::accumulate(&tnow[INDEX2D(1, j)], &tnow[INDEX2D(IMAX-1, j)], temperature);
  }
  std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> runtime = std::chrono::duration_cast<std::chrono::duration<double>>(t_end-t_start);
  std::cout << "Final Temperature: " << temperature << " Expected: " << expected << std::endl;
  std::cout << "Time Elapsed: " << runtime.count() << "s" << std::endl;
  return EXIT_SUCCESS;
}

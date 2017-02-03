#include <iostream> 
#include <vector>
#include <chrono>
#include <numeric>

#define TIMESTEPS 100

#define STENCIL_RADIUS 1

#define IMAX 100

#define HOTCORNER_IMAX 25

int main() {
  const double nu = 0.30;
  const double sigma = 0.2;
  const double width = 2.0;
  const double dx = width / (IMAX-1);
  const double dt = sigma * dx * dx / nu;
  std::vector<double> tnow(IMAX);
  std::vector<double> tnext(IMAX);
  for (int i = 0; i < HOTCORNER_IMAX; ++i) {
    tnow[i] = 2.0;
  }
  for (int i = HOTCORNER_IMAX; i < IMAX; ++i) {
    tnow[i] = 1.0;
  }
  std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
  // Calculate initial (inner) temperature
  double expected =((HOTCORNER_IMAX-1)+(IMAX-2)) * 1.0;
  double temperature = std::accumulate(&tnow[1], &tnow[IMAX-1], 0.0);
  std::cout << "Initial Temperature: " << temperature << " Expected: " << expected << std::endl;
  for (int ts = 0; ts < TIMESTEPS; ++ts) {
    // Diffusion
    for (int i = 1; i < IMAX-1; ++i) {
      tnext[i] = tnow[i]+(nu * dt / (dx * dx)) * (tnow[i-1] - 2.0 * tnow[i]+tnow[i+1]);
    }
    // Reflective Boundary Condition
    tnext[0] = tnext[1];
    tnext[IMAX-1] = tnext[IMAX-2];
    std::swap(tnow, tnext); 
  }
  for (auto i: tnow) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  temperature = std::accumulate(&tnow[1], &tnow[IMAX-1], 0.0);
  std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> runtime = std::chrono::duration_cast<std::chrono::duration<double>>(t_end-t_start);
  std::cout << "Final Temperature: " << temperature << " Expected: " << expected << std::endl;
  std::cout << "Time Elapsed: " << runtime.count() << "s" << std::endl;
  return EXIT_SUCCESS;
}

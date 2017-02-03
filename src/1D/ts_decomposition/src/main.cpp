#include <iostream> 
#include <vector>
#include <chrono>
#include <numeric>

#define TIMESTEPS 100

#define STENCIL_RADIUS 1

#define IMAX 100

#define HOTCORNER_IMAX 25

const double nu = 0.30;
const double sigma = 0.2;
const double width = 2;
const double dx = width / (IMAX - 1);
const double dt = sigma * dx * dx / nu;
const double cx = nu * dt / (dx * dx);

const int t_threshold = 20;
const int i_threshold = 25;

std::vector<double> talpha(IMAX);
std::vector<double> tbeta(IMAX);

double *state[2] = {talpha.data(), tbeta.data()};
#define STATE_NOW (state[ts &1])
#define STATE_NEXT (state[(ts + 1) &1])
#define STATE_INITIAL talpha
#define STATE_FINAL (state[(TIMESTEPS)&1])

inline void BaseTrapezoid(int t0, int t1,
                          int i0, int del_i0,
                          int i1, int del_i1) {
  for (int ts = t0; ts < t1; ++ts) {
    for (int i = i0; i < i1; ++i) {
      // Diffusion
      STATE_NEXT[i] = STATE_NOW[i] + (nu * dt / (dx * dx)) * (STATE_NOW[i-1] - 2.0 * STATE_NOW[i] + STATE_NOW[i+1]);
      // Reflective Boundary Condition
      if (i == 1) {
        STATE_NEXT[0] = STATE_NEXT[i];
      } else if (i == IMAX-2) {
        STATE_NEXT[IMAX-1] = STATE_NEXT[i];
      }
    }
    // Slide along trapzoid
    i0 += del_i0;
    i1 += del_i1;
  }
}

void RecursiveTrapezoid(int t0, int t1,
                        int i0, int del_i0,
                        int i1, int del_i1) { 
  int t_span = t1 - t0;
  int i_span = i1 - i0;
  if (t_span > 1) { // Trapezoid slope becomes meaningless over a single timestep
    if (i_span > i_threshold && i_span > 2 * STENCIL_RADIUS * t_span * 2) {
      // Major trapezoids (x2)
      int half_i_span = i_span/2;
      RecursiveTrapezoid(t0, t1,
                         i0, STENCIL_RADIUS,
                         i0 + half_i_span, -STENCIL_RADIUS);
      RecursiveTrapezoid(t0, t1,
                         i0 + half_i_span, STENCIL_RADIUS,
                         i1, -STENCIL_RADIUS);
      // Filler trapezoids (x3, possibly empty)
      // Left
      RecursiveTrapezoid(t0, t1,
                         i0, del_i0,
                         i0, STENCIL_RADIUS);
      // Center
      RecursiveTrapezoid(t0, t1,
                         i0+half_i_span, -STENCIL_RADIUS,
                         i0+half_i_span, STENCIL_RADIUS);
      // Right
      RecursiveTrapezoid(t0, t1,
                         i1, -STENCIL_RADIUS,
                         i1, del_i1);
      return;
    }
    if (t_span > t_threshold) {
      int half_t_span = t_span/2;
      RecursiveTrapezoid(t0, t0 + half_t_span,
                         i0, del_i0,
                         i1, del_i1);
      RecursiveTrapezoid(t0 + half_t_span, t1,
                         i0 + del_i0 * half_t_span , del_i0, 
                         i1 + del_i1 * half_t_span, del_i1);
      return;
    }
  }
  BaseTrapezoid(t0, t1, i0, del_i0, i1, del_i1);
}

int main() {
  for (int i = 0; i < HOTCORNER_IMAX; ++i) {
    STATE_INITIAL[i] = 2.0;
  }
  for (int i = HOTCORNER_IMAX; i < IMAX; ++i) {
    STATE_INITIAL[i] = 1.0;
  }
  std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
  // Calculate initial (inner) temperature
  double expected = ((HOTCORNER_IMAX - 1) + (IMAX - 2)) * 1.0;
  double temperature = std::accumulate(&STATE_INITIAL[1], &STATE_INITIAL[IMAX-1], 0.0);
  std::cout << "Initial Temperature: " << temperature << " Expected: " << expected << std::endl;
  RecursiveTrapezoid(0, TIMESTEPS, 1, 0, IMAX-1, 0);
  temperature = std::accumulate(&STATE_FINAL[1], &STATE_FINAL[IMAX-1], 0.0);
  std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> runtime = std::chrono::duration_cast<std::chrono::duration<double>>(t_end-t_start);
  std::cout << "Final Temperature: " << temperature << " Expected: " << expected << std::endl;
  std::cout << "Time Elapsed: " << runtime.count() << "s" << std::endl;
  return EXIT_SUCCESS;
}

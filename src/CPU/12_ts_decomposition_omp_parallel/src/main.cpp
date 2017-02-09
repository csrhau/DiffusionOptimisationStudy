#include <iostream> 
#include <vector>
#include <chrono>
#include <numeric>

#define TIMESTEPS 600

#define STENCIL_RADIUS 1

#define IMAX 800
#define JMAX 800
#define KMAX 800

#define HOTCORNER_IMAX 25
#define HOTCORNER_JMAX 25
#define HOTCORNER_KMAX 25

#define INDEX3D(i, j, k) ((((k) * (JMAX) * (IMAX)) + ((j) * (IMAX)) + (i)))

const double nu = 0.05;
const double sigma = 0.25;
const double width = 2;
const double height = 2;
const double dx = width / (IMAX-1);
const double dy = height / (JMAX-1);
const double dz = height / (KMAX-1);
const double dt = sigma * dx * dy * dz / nu;

const int t_threshold = 20;
const int i_threshold = 10;
const int j_threshold = 10;
const int k_threshold = 10;

std::vector<double> talpha(IMAX * JMAX * KMAX);
std::vector<double> tbeta(IMAX * JMAX * KMAX);

double *state[2] = {talpha.data(), tbeta.data()};
#define STATE_NOW (state[ts &1])
#define STATE_NEXT (state[(ts + 1) &1])
#define STATE_INITIAL talpha
#define STATE_FINAL (state[(TIMESTEPS)&1])

inline void BaseTrapezoid(int t0, int t1,
                          int i0, int del_i0, int i1, int del_i1,
                          int j0, int del_j0, int j1, int del_j1,
                          int k0, int del_k0, int k1, int del_k1) {
  for (int ts = t0; ts < t1; ++ts) {
    for (int k = k0; k < k1; ++k) {
      for (int j = j0; j < j1; ++j) {
        #pragma simd
        for (int i = i0; i < i1; ++i) {
          // Diffusion
          STATE_NEXT[INDEX3D(i, j, k)] = STATE_NOW[INDEX3D(i, j, k)] + (nu * dt / (dx * dx)) * (STATE_NOW[INDEX3D(i-1, j, k)]-2.0* STATE_NOW[INDEX3D(i, j, k)] + STATE_NOW[INDEX3D(i+1, j, k)])
                                                                     + (nu * dt / (dy * dy)) * (STATE_NOW[INDEX3D(i, j-1, k)]-2.0* STATE_NOW[INDEX3D(i, j, k)] + STATE_NOW[INDEX3D(i, j+1, k)])
                                                                     + (nu * dt / (dz * dz)) * (STATE_NOW[INDEX3D(i, j, k-1)]-2.0* STATE_NOW[INDEX3D(i, j, k)] + STATE_NOW[INDEX3D(i, j, k+1)]);
          // Reflective Boundary Conditions
          if (i==1) {
            STATE_NEXT[INDEX3D(i-1, j, k)] = STATE_NEXT[INDEX3D(i, j, k)];
          } else if (i == IMAX - 2) {
            STATE_NEXT[INDEX3D(i+1, j, k)] = STATE_NEXT[INDEX3D(i, j, k)];
          } 
          if (j == 1) {
            STATE_NEXT[INDEX3D(i, j-1, k)] = STATE_NEXT[INDEX3D(i, j, k)];
          } else if (j == JMAX - 2) {
            STATE_NEXT[INDEX3D(i, j+1, k)] = STATE_NEXT[INDEX3D(i, j, k)];
          } 
          if (k == 1) {
            STATE_NEXT[INDEX3D(i, j, k-1)] = STATE_NEXT[INDEX3D(i, j, k)];
          } else if (k == KMAX - 2) {
            STATE_NEXT[INDEX3D(i, j, k+1)] = STATE_NEXT[INDEX3D(i, j, k)];
          } 
        }
      }
    }
    i0 += del_i0;
    i1 += del_i1;
    j0 += del_j0;
    j1 += del_j1;
    k0 += del_k0;
    k1 += del_k1;
  }
}

inline void RecursiveTrapezoid(int t0, int t1,
                               int i0, int del_i0, int i1, int del_i1,
                               int j0, int del_j0, int j1, int del_j1,
                               int k0, int del_k0, int k1, int del_k1) {
  int t_span = t1 - t0;
  int i_span = i1 - i0;
  int j_span = j1 - j0;
  int k_span = k1 - k0;
  int split_threshold = 2 * STENCIL_RADIUS * t_span * 2;
  if (t_span > 1) { // Trapezoid slope becomes meaningless over a single timestep
    if (i_span > i_threshold && i_span >= j_span && i_span >= k_span && i_span > split_threshold) {
      int half_i_span = i_span/2;
      // Core trapezoids (x2)
      #pragma omp task 
      RecursiveTrapezoid(t0, t1,
                         i0, STENCIL_RADIUS,
                         i0 + half_i_span, -STENCIL_RADIUS,
                         j0, del_j0, j1, del_j1,
                         k0, del_k0, k1, del_k1);
      #pragma omp task 
      RecursiveTrapezoid(t0, t1,
                         i0 + half_i_span, STENCIL_RADIUS,
                         i1, -STENCIL_RADIUS,
                         j0, del_j0, j1, del_j1,
                         k0, del_k0, k1, del_k1);
      #pragma omp taskwait
      // Filler trapezoids (x3, possibly empty)
      #pragma omp task 
      RecursiveTrapezoid(t0, t1,
                         i0, del_i0,
                         i0, STENCIL_RADIUS,
                         j0, del_j0, j1, del_j1, 
                         k0, del_k0, k1, del_k1);
      #pragma omp task 
      RecursiveTrapezoid(t0, t1,
                         i0+half_i_span, -STENCIL_RADIUS,
                         i0+half_i_span, STENCIL_RADIUS,
                         j0, del_j0, j1, del_j1,
                         k0, del_k0, k1, del_k1);
      #pragma omp task 
      RecursiveTrapezoid(t0, t1,
                         i1, -STENCIL_RADIUS,
                         i1, del_i1,
                         j0, del_j0, j1, del_j1,
                         k0, del_k0, k1, del_k1);
      #pragma omp taskwait
      return;
    } 
    if (j_span > j_threshold && j_span >= k_span && j_span > split_threshold) {
      int half_j_span = j_span/2;
      // Core trapezoids (x2)
      #pragma omp task 
      RecursiveTrapezoid(t0, t1,
                         i0, del_i0, i1, del_i1,
                         j0, STENCIL_RADIUS,
                         j0 + half_j_span, -STENCIL_RADIUS,
                         k0, del_k0, k1, del_k1);
      #pragma omp task 
      RecursiveTrapezoid(t0, t1,
                         i0, del_i0, i1, del_i1,
                         j0 + half_j_span, STENCIL_RADIUS,
                         j1, -STENCIL_RADIUS,
                         k0, del_k0, k1, del_k1);
      #pragma omp taskwait
      // Filler trapezoids (x3, possibly empty)
      #pragma omp task 
      RecursiveTrapezoid(t0, t1,
                         i0, del_i0, i1, del_i1,
                         j0, del_j0,
                         j0, STENCIL_RADIUS,
                         k0, del_k0, k1, del_k1);
      #pragma omp task 
      RecursiveTrapezoid(t0, t1,
                         i0, del_i0, i1, del_i1,
                         j0+half_j_span, -STENCIL_RADIUS,
                         j0+half_j_span, STENCIL_RADIUS,
                         k0, del_k0, k1, del_k1);
      #pragma omp task 
      RecursiveTrapezoid(t0, t1,
                         i0, del_i0, i1, del_i1,
                         j1, -STENCIL_RADIUS,
                         j1, del_j1,
                         k0, del_k0, k1, del_k1);
      #pragma omp taskwait
      return;
    } 
    if (k_span > k_threshold && k_span > split_threshold) {
      int half_k_span = k_span/2;
      // Core trapezoids (x2)
      #pragma omp task 
      RecursiveTrapezoid(t0, t1,
                         i0, del_i0, i1, del_i1,
                         j0, del_j0, j1, del_j1,
                         k0, STENCIL_RADIUS,
                         k0 + half_k_span, -STENCIL_RADIUS);
      #pragma omp task 
      RecursiveTrapezoid(t0, t1,
                         i0, del_i0, i1, del_i1,
                         j0, del_j0, j1, del_j1,
                         k0 + half_k_span, STENCIL_RADIUS,
                         k1, -STENCIL_RADIUS);
      #pragma omp taskwait
      // Filler trapezoids (x3, possibly empty)
      #pragma omp task 
      RecursiveTrapezoid(t0, t1,
                         i0, del_i0, i1, del_i1,
                         j0, del_j0, j1, del_j1,
                         k0, del_k0,
                         k0, STENCIL_RADIUS);
      #pragma omp task 
      RecursiveTrapezoid(t0, t1,
                         i0, del_i0, i1, del_i1,
                         j0, del_j0, j1, del_j1,
                         k0+half_k_span, -STENCIL_RADIUS,
                         k0+half_k_span, STENCIL_RADIUS);
      #pragma omp task 
      RecursiveTrapezoid(t0, t1,
                         i0, del_i0, i1, del_i1,
                         j0, del_j0, j1, del_j1,
                         k1, -STENCIL_RADIUS,
                         k1, del_k1);
      #pragma omp taskwait
      return;
    }
    if (t_span > t_threshold) {
      int half_t_span = t_span / 2;
      RecursiveTrapezoid(t0, t0 + half_t_span,
                         i0, del_i0, i1, del_i1,
                         j0, del_j0, j1, del_j1,
                         k0, del_k0, k1, del_k1);
      RecursiveTrapezoid(t0 + half_t_span, t1,
                         i0 + del_i0 * half_t_span , del_i0, 
                         i1 + del_i1 * half_t_span, del_i1,
                         j0 + del_j0 * half_t_span , del_j0, 
                         j1 + del_j1 * half_t_span, del_j1,
                         k0 + del_k0 * half_t_span , del_k0, 
                         k1 + del_k1 * half_t_span, del_k1);
      return;
    }
  }
  BaseTrapezoid(t0, t1,
                i0, del_i0, i1, del_i1,
                j0, del_j0, j1, del_j1,
                k0, del_k0, k1, del_k1);
}

int main() {
  for (int k = 0; k < KMAX; ++k) {
    for (int j = 0; j < JMAX; ++j) {
      for (int i = 0; i < IMAX; ++i) {
        if (i < HOTCORNER_IMAX && j < HOTCORNER_JMAX && k < HOTCORNER_KMAX) {
          STATE_INITIAL[INDEX3D(i, j, k)] = 2.0;
        } else {
          STATE_INITIAL[INDEX3D(i, j, k)] = 1.0;
        }
      }
    }
  }
  std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
  // Calculate initial (inner) temperature
  const unsigned long all_cells = (IMAX-2) * (JMAX-2) * (KMAX-2);
  const unsigned long hot_cells = (HOTCORNER_IMAX-1) * (HOTCORNER_JMAX-1) * (HOTCORNER_KMAX-1);
  double expected = hot_cells * 2.0 + (all_cells-hot_cells) * 1.0;
  double temperature = 0.0;
  #pragma omp parallel for reduction(+:temperature)
  for (int k = 1; k < KMAX-1; ++k) {
    for (int j = 1; j < JMAX-1; ++j) {
      temperature = std::accumulate(&STATE_INITIAL[INDEX3D(1, j, k)], &STATE_INITIAL[INDEX3D(IMAX-1, j, k)], temperature);
    }
  }
  std::cout << "Initial Temperature: " << temperature << " Expected: " << expected << std::endl;
  std::chrono::steady_clock::time_point t_sim_start = std::chrono::steady_clock::now();
  #pragma omp parallel
  #pragma omp single nowait
  {
    RecursiveTrapezoid(0, TIMESTEPS, 1, 0, IMAX-1, 0, 1, 0, JMAX-1, 0, 1, 0, KMAX-1, 0);
  }
  std::chrono::steady_clock::time_point t_sim_end = std::chrono::steady_clock::now();
  temperature = 0.0;
  #pragma omp parallel for reduction(+:temperature)
  for (int k = 1; k < KMAX-1; ++k) {
    for (int j = 1; j < JMAX-1; ++j) {
      temperature = std::accumulate(&STATE_FINAL[INDEX3D(1, j, k)], &STATE_FINAL[INDEX3D(IMAX-1, j, k)], temperature);
    }
  }
  std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> runtime = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start);
  std::chrono::duration<double> sim_runtime = std::chrono::duration_cast<std::chrono::duration<double>>(t_sim_end-t_sim_start);
  std::cout << "Final Temperature: " << temperature << " Expected: " << expected << std::endl;
  std::cout << "Time Elapsed (simulation): " << sim_runtime.count() << "s" << std::endl;
  std::cout << "Time Elapsed (total): " << runtime.count() << "s" << std::endl;
  return EXIT_SUCCESS;
}

#include <iostream> 
#include <vector>
#include <chrono>
#include <numeric>
#include <omp.h>

#define TIMESTEPS 360

#define IMAX 800
#define JMAX 800
#define KMAX 800

#define ISTRIPSPAN 32
#define JSTRIPSPAN 16
#define KSTRIPSPAN 16

#define HOTCORNER_IMAX 25
#define HOTCORNER_JMAX 25
#define HOTCORNER_KMAX 25

#define INDEX3D(i, j, k) ((((k) * (JMAX) * (IMAX)) + ((j) * (IMAX)) + (i)))

int main() {
  const double nu = 0.05;
  const double sigma = 0.25;
  const double width = 2;
  const double height = 2;
  const double dx = width / (IMAX-1);
  const double dy = height / (JMAX-1);
  const double dz = height / (KMAX-1);
  const double dt = sigma * dx * dy * dz / nu;
  std::vector<double> tnow(IMAX * JMAX * KMAX);
  std::vector<double> tnext(IMAX * JMAX * KMAX);
  const unsigned int kstrips = (KMAX + KSTRIPSPAN - 1) / KSTRIPSPAN;
  const unsigned int jstrips = (JMAX + JSTRIPSPAN - 1) / JSTRIPSPAN;
  const unsigned int istrips = (IMAX + ISTRIPSPAN - 1) / ISTRIPSPAN;
  #pragma omp parallel for
  for (int ks = 0; ks < kstrips; ++ks) {
    const unsigned int kmin = ks * KSTRIPSPAN;
    const unsigned int kmax = (ks == (kstrips - 1) ? KMAX : (ks + 1) * KSTRIPSPAN);
    for (int js = 0; js < jstrips; ++js) {
      const unsigned int jmin = js * JSTRIPSPAN;
      const unsigned int jmax = (js == (jstrips - 1) ? JMAX : (js + 1) * JSTRIPSPAN);
      for (int is = 0; is < istrips; ++is) {
        const unsigned int imin = is * ISTRIPSPAN;
        const unsigned int imax = (is == (istrips - 1) ? IMAX : (is + 1) * ISTRIPSPAN);
        for (int k = kmin; k < kmax; ++k) {
          for (int j = jmin; j < jmax; ++j) {
            for (int i = imin; i < imax; ++i) {
              if (i < HOTCORNER_IMAX && j < HOTCORNER_JMAX && k < HOTCORNER_KMAX) {
                tnow[INDEX3D(i, j, k)] = 2.0;
              } else {
                tnow[INDEX3D(i, j, k)] = 1.0;
              }
            }
          }
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
      temperature = std::accumulate(&tnow[INDEX3D(1, j, k)], &tnow[INDEX3D(IMAX-1, j, k)], temperature);
    }
  }
  std::cout << "Initial Temperature: " << temperature << " Expected: " << expected << std::endl;
  for (int ts = 0; ts < TIMESTEPS; ++ts) {
    // Diffusion
    #pragma omp parallel for schedule(dynamic, 1) collapse(3)
    for (int ks = 0; ks < kstrips; ++ks) {
      for (int js = 0; js < jstrips; ++js) {
        for (int is = 0; is < istrips; ++is) {
          const unsigned int kmin = (ks == 0 ? 1 : ks * KSTRIPSPAN);
          const unsigned int kmax = (ks == (kstrips - 1) ? KMAX-1 : (ks + 1) * KSTRIPSPAN);
          const unsigned int jmin = (js == 0 ? 1 : js * JSTRIPSPAN);
          const unsigned int jmax = (js == (jstrips - 1) ? JMAX-1 : (js + 1) * JSTRIPSPAN);
          const unsigned int imin = (is == 0 ? 1 : is * ISTRIPSPAN);
          const unsigned int imax = (is == (istrips - 1) ? IMAX-1 : (is + 1) * ISTRIPSPAN);
          for (int k = kmin; k < kmax; ++k) {
            for (int j = jmin; j < jmax; ++j) {
              for (int i = imin; i < imax; ++i) {
                tnext[INDEX3D(i, j, k)] = tnow[INDEX3D(i, j, k)] + (nu * dt / (dx * dx)) * (tnow[INDEX3D(i-1, j, k)]-2.0* tnow[INDEX3D(i, j, k)] + tnow[INDEX3D(i+1, j, k)])
                                                                 + (nu * dt / (dy * dy)) * (tnow[INDEX3D(i, j-1, k)]-2.0* tnow[INDEX3D(i, j, k)] + tnow[INDEX3D(i, j+1, k)])
                                                                 + (nu * dt / (dz * dz)) * (tnow[INDEX3D(i, j, k-1)]-2.0* tnow[INDEX3D(i, j, k)] + tnow[INDEX3D(i, j, k+1)]);
              }
            }
          }
        }
      }
    }
    // Reflective Boundary Condition
    #pragma omp parallel for
    for (int k = 1; k < KMAX-1; ++k) {
      for (int j = 1; j < JMAX-1; ++j) {
        tnext[INDEX3D(0, j, k)] = tnext[INDEX3D(1, j, k)];
        tnext[INDEX3D(IMAX-1, j, k)] = tnext[INDEX3D(IMAX-2, j, k)];
      }
      for (int i = 1; i < IMAX-1; ++i) {
        tnext[INDEX3D(i, 0, k)] = tnext[INDEX3D(i, 1, k)];
        tnext[INDEX3D(i, JMAX-1, k)] = tnext[INDEX3D(i, JMAX-2, k)]; }
    }
    #pragma omp parallel for
    for (int j = 1; j < JMAX-1; ++j) {
      for (int i = 1; i < IMAX-1; ++i) {
        tnext[INDEX3D(i, j, 0)] = tnext[INDEX3D(i, j, 1)];
        tnext[INDEX3D(i, j, KMAX-1)] = tnext[INDEX3D(i, j, KMAX-2)];
      }
    }
    std::swap(tnow, tnext); 
  }
  temperature = 0.0;
  #pragma omp parallel for reduction(+:temperature)
  for (int k = 1; k < KMAX-1; ++k) {
    for (int j = 1; j < JMAX-1; ++j) {
      temperature = std::accumulate(&tnow[INDEX3D(1, j, k)], &tnow[INDEX3D(IMAX-1, j, k)], temperature);
    }
  }
  std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> runtime = std::chrono::duration_cast<std::chrono::duration<double>>(t_end-t_start);
  std::cout << "Final Temperature: " << temperature << " Expected: " << expected << std::endl;
  std::cout << "Time Elapsed: " << runtime.count() << "s" << std::endl;
  return EXIT_SUCCESS;
}

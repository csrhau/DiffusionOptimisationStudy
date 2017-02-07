#include <iostream> 
#include <vector>
#include <chrono>
#include <numeric>
#include <cilk/cilk.h>

#define TIMESTEPS 600

#define IMAX 800
#define JMAX 800
#define KMAX 800 

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

  double (&tnow_arr)[KMAX][JMAX][IMAX] = *reinterpret_cast<double (*) [KMAX][JMAX][IMAX]>(tnow.data());
  double (&tnext_arr)[KMAX][JMAX][IMAX] = *reinterpret_cast<double (*) [KMAX][JMAX][IMAX]>(tnext.data());
  for (int k = 0; k < KMAX; ++k) {
    for (int j = 0; j < JMAX; ++j) {
      for (int i = 0; i < IMAX; ++i) {
        if (i < HOTCORNER_IMAX && j < HOTCORNER_JMAX && k < HOTCORNER_KMAX) {
          tnow[INDEX3D(i, j, k)] = 2.0;
        } else {
          tnow[INDEX3D(i, j, k)] = 1.0;
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
  {
    double (&tnow_arr)[KMAX][JMAX][IMAX] = *reinterpret_cast<double (*) [KMAX][JMAX][IMAX]>(tnow.data());
    temperature = __sec_reduce_add(tnow_arr[1:KMAX-2][1:JMAX-2][1:IMAX-2]);
  }
  std::cout << "Initial Temperature: " << temperature << " Expected: " << expected << std::endl;
  std::chrono::steady_clock::time_point t_sim_start = std::chrono::steady_clock::now();
  for (int ts = 0; ts < TIMESTEPS; ++ts) {
    double (&tnow_arr)[KMAX][JMAX][IMAX] = *reinterpret_cast<double (*) [KMAX][JMAX][IMAX]>(tnow.data());
    double (&tnext_arr)[KMAX][JMAX][IMAX] = *reinterpret_cast<double (*) [KMAX][JMAX][IMAX]>(tnext.data());
    // Diffusion
    cilk_for (int k = 1; k < KMAX - 1; ++k) {
      tnext_arr[k][1:JMAX-2][1:IMAX-2] = tnow_arr[k][1:JMAX-2][1:IMAX-2]
                                       + (nu * dt / (dx * dx)) * (tnow_arr[k][1:JMAX-2][0:IMAX-2] - 2.0*tnow_arr[k][1:JMAX-2][1:IMAX-2] + tnow_arr[k][1:JMAX-2][2:IMAX-2])
                                       + (nu * dt / (dy * dy)) * (tnow_arr[k][0:JMAX-2][1:IMAX-2] - 2.0*tnow_arr[k][1:JMAX-2][1:IMAX-2] + tnow_arr[k][2:JMAX-2][1:IMAX-2])
                                       + (nu * dt / (dz * dz)) * (tnow_arr[k-1][1:JMAX-2][1:IMAX-2] - 2.0*tnow_arr[k][1:JMAX-2][1:IMAX-2] + tnow_arr[k+1][1:JMAX-2][1:IMAX-2]);
    }
    // Reflective Boundary Condition
    auto iboundary = [&]() {
      tnext_arr[1:KMAX-2][1:JMAX-2][0] = tnext_arr[1:KMAX-2][1:JMAX-2][1];
      tnext_arr[1:KMAX-2][1:JMAX-2][IMAX-1] = tnext_arr[1:KMAX-2][1:JMAX-2][IMAX-2];
    };
    cilk_spawn iboundary();

    auto jboundary = [&]() {
      tnext_arr[1:KMAX-2][0][1:IMAX-2] = tnext_arr[1:KMAX-2][1][1:IMAX-2];
      tnext_arr[1:KMAX-2][JMAX-1][1:IMAX-2] = tnext_arr[1:KMAX-2][JMAX-2][1:IMAX-2];
    };
    cilk_spawn jboundary();
    tnext_arr[0][1:JMAX-2][1:IMAX-2] = tnext_arr[1][1:JMAX-2][1:IMAX-2];
    tnext_arr[KMAX-1][1:JMAX-2][1:IMAX-2] = tnext_arr[KMAX-2][1:JMAX-2][1:IMAX-2];
    std::swap(tnow, tnext); 
  }
  std::chrono::steady_clock::time_point t_sim_end = std::chrono::steady_clock::now();
  temperature = 0.0;
  {
    double (&tnow_arr)[KMAX][JMAX][IMAX] = *reinterpret_cast<double (*) [KMAX][JMAX][IMAX]>(tnow.data());
    temperature = __sec_reduce_add(tnow_arr[1:KMAX-2][1:JMAX-2][1:IMAX-2]);
  }
  std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> runtime = std::chrono::duration_cast<std::chrono::duration<double>>(t_end-t_start);
  std::chrono::duration<double> sim_runtime = std::chrono::duration_cast<std::chrono::duration<double>>(t_sim_end-t_sim_start);
  std::cout << "Final Temperature: " << temperature << " Expected: " << expected << std::endl;
  std::cout << "Time Elapsed (simulation): " << sim_runtime.count() << "s" << std::endl;
  std::cout << "Time Elapsed (total): " << runtime.count() << "s" << std::endl;
  return EXIT_SUCCESS;
}

#include "simulation.h"

#include "kernels.cuh"

#include <stdio.h>

#include "common.h"

void RecursiveTrapezoid(double *state[2],
                        double cx, double cy, double cz,
                        int t0, int t1,
                        int i0, int del_i0, int i1, int del_i1, int imax,
                        int j0, int del_j0, int j1, int del_j1, int jmax,
                        int k0, int del_k0, int k1, int del_k1, int kmax) {
  int t_span = t1 - t0;
  int i_span = i1 - i0;
  int j_span = j1 - j0;
  int k_span = k1 - k0;
  if ((i_span == 0 && del_i0 == del_i1)
   || (j_span == 0 && del_j0 == del_j1)
   || (k_span == 0 && del_k0 == del_k1)) { 
    return;
  }
  int split_threshold = 2 * STENCIL_RADIUS * t_span * 2;
  if (t_span > 1) { // Trapezoid slope becomes meaningless over a single timestep
    if (i_span > I_THRESHOLD && i_span >= j_span && i_span >= k_span && i_span > split_threshold) {
      int half_i_span = i_span/2;
      // Core trapezoids (x2)
      RecursiveTrapezoid(state,
                         cx, cy, cz,
                         t0, t1,
                         i0, STENCIL_RADIUS, i0 + half_i_span, -STENCIL_RADIUS, imax,
                         j0, del_j0, j1, del_j1, jmax,
                         k0, del_k0, k1, del_k1, kmax);
      RecursiveTrapezoid(state,
                         cx, cy, cz,
                         t0, t1,
                         i0 + half_i_span, STENCIL_RADIUS, i1, -STENCIL_RADIUS, imax,
                         j0, del_j0, j1, del_j1, jmax,
                         k0, del_k0, k1, del_k1, kmax);
      // Filler trapezoids (x3, possibly empty)
      RecursiveTrapezoid(state,
                         cx, cy, cz,
                         t0, t1,
                         i0, del_i0, i0, STENCIL_RADIUS, imax,
                         j0, del_j0, j1, del_j1, jmax,
                         k0, del_k0, k1, del_k1, kmax);
      RecursiveTrapezoid(state,
                         cx, cy, cz,
                         t0, t1,
                         i0+half_i_span, -STENCIL_RADIUS, i0+half_i_span, STENCIL_RADIUS, imax,
                         j0, del_j0, j1, del_j1, jmax,
                         k0, del_k0, k1, del_k1, kmax);
      RecursiveTrapezoid(state,
                         cx, cy, cz,
                         t0, t1,
                         i1, -STENCIL_RADIUS, i1, del_i1, imax,
                         j0, del_j0, j1, del_j1, jmax,
                         k0, del_k0, k1, del_k1, kmax);
      return;
    } 
    if (j_span > J_THRESHOLD && j_span >= k_span && j_span > split_threshold) {
      int half_j_span = j_span/2;
      // Core trapezoids (x2)
      RecursiveTrapezoid(state,
                         cx, cy, cz,
                         t0, t1,
                         i0, del_i0, i1, del_i1, imax,
                         j0, STENCIL_RADIUS, j0 + half_j_span, -STENCIL_RADIUS, jmax,
                         k0, del_k0, k1, del_k1, kmax);
      RecursiveTrapezoid(state,
                         cx, cy, cz,
                         t0, t1,
                         i0, del_i0, i1, del_i1, imax,
                         j0 + half_j_span, STENCIL_RADIUS, j1, -STENCIL_RADIUS, jmax,
                         k0, del_k0, k1, del_k1, kmax);
      // Filler trapezoids (x3, possibly empty)
      RecursiveTrapezoid(state,
                         cx, cy, cz,
                         t0, t1,
                         i0, del_i0, i1, del_i1, imax,
                         j0, del_j0, j0, STENCIL_RADIUS, jmax,
                         k0, del_k0, k1, del_k1, kmax);
      RecursiveTrapezoid(state,
                         cx, cy, cz,
                         t0, t1,
                         i0, del_i0, i1, del_i1, imax,
                         j0+half_j_span, -STENCIL_RADIUS, j0+half_j_span, STENCIL_RADIUS, jmax,
                         k0, del_k0, k1, del_k1, kmax);
      RecursiveTrapezoid(state,
                         cx, cy, cz,
                         t0, t1,
                         i0, del_i0, i1, del_i1, imax,
                         j1, -STENCIL_RADIUS, j1, del_j1, jmax,
                         k0, del_k0, k1, del_k1, kmax);
      return;
    } 
    if (k_span > K_THRESHOLD && k_span > split_threshold) {
      int half_k_span = k_span/2;
      // Core trapezoids (x2)
      RecursiveTrapezoid(state,
                         cx, cy, cz,
                         t0, t1,
                         i0, del_i0, i1, del_i1, imax,
                         j0, del_j0, j1, del_j1, jmax,
                         k0, STENCIL_RADIUS, k0 + half_k_span, -STENCIL_RADIUS, kmax);
      RecursiveTrapezoid(state,
                         cx, cy, cz,
                         t0, t1,
                         i0, del_i0, i1, del_i1, imax,
                         j0, del_j0, j1, del_j1, jmax,
                         k0 + half_k_span, STENCIL_RADIUS, k1, -STENCIL_RADIUS, kmax);
      // Filler trapezoids (x3, possibly empty)
      RecursiveTrapezoid(state,
                         cx, cy, cz,
                         t0, t1,
                         i0, del_i0, i1, del_i1, imax,
                         j0, del_j0, j1, del_j1, jmax, 
                         k0, del_k0, k0, STENCIL_RADIUS, kmax);
      RecursiveTrapezoid(state,
                         cx, cy, cz,
                         t0, t1,
                         i0, del_i0, i1, del_i1, imax,
                         j0, del_j0, j1, del_j1, jmax,
                         k0+half_k_span, -STENCIL_RADIUS, k0+half_k_span, STENCIL_RADIUS, kmax);
      RecursiveTrapezoid(state,
                         cx, cy, cz,
                         t0, t1,
                         i0, del_i0, i1, del_i1, imax,
                         j0, del_j0, j1, del_j1, jmax,
                         k1, -STENCIL_RADIUS, kmax,
                         k1, del_k1);
      return;
    }
    if (t_span > T_THRESHOLD) {
      int half_t_span = t_span / 2;
      RecursiveTrapezoid(state,
                         cx, cy, cz,
                         t0, t0 + half_t_span,
                         i0, del_i0, i1, del_i1, imax,
                         j0, del_j0, j1, del_j1, jmax,
                         k0, del_k0, k1, del_k1, kmax);
      RecursiveTrapezoid(state,
                         cx, cy, cz,
                         t0 + half_t_span, t1,
                         i0 + del_i0 * half_t_span , del_i0, i1 + del_i1 * half_t_span, del_i1, imax,
                         j0 + del_j0 * half_t_span , del_j0, j1 + del_j1 * half_t_span, del_j1, jmax,
                         k0 + del_k0 * half_t_span , del_k0, k1 + del_k1 * half_t_span, del_k1, kmax);
      return;
    }
  }
  BaseTrapezoid(state,
                cx, cy, cz,
                t0, t1,
                i0, del_i0, i1, del_i1, imax,
                j0, del_j0, j1, del_j1, jmax, 
                k0, del_k0, k1, del_k1, kmax);
}



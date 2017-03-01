#pragma once

void RecursiveTrapezoid(double *state[2],
                        double cx, double cy, double cz,
                        int t0, int t1,
                        int i0, int del_i0, int i1, int del_i1,
                        int j0, int del_j0, int j1, int del_j1,
                        int k0, int del_k0, int k1, int del_k1);

void BaseTrapezoid(double *state[2],
                   double cx, double cy, double cz,
                   int t0, int t1,
                   int i0, int del_i0, int i1, int del_i1,
                   int j0, int del_j0, int j1, int del_j1,
                   int k0, int del_k0, int k1, int del_k1);

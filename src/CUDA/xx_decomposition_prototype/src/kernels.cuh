#pragma once
// vim: set ft=cpp:

__global__ void DiffuseReflectKnl(
    double *__restrict__ tnow,
    double *__restrict__ tnext,
    int ts,
    int i0, int i1, int imax,
    int j0, int j1, int jmax,
    int k0, int k1, int kmax,
    double cx,
    double cy,
    double cz
);


__global__ void PrintKnl(
    double *__restrict__ tnow,
    double *__restrict__ tnext,
    int ts,
    int i0, int i1, int imax,
    int j0, int j1, int jmax,
    int k0, int k1, int kmax,
    double cx,
    double cy,
    double cz
);

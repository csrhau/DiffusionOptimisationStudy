#pragma once
/* vim: set ft=cpp : */ 

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "common.cuh"

__global__ void DiffuseKnl(double *__restrict__ tnow,
                           double *__restrict__ tnext,
                           double cx,
                           double cy,
                           double cz);

__global__ void ReflectIKnl(double *tnext);
__global__ void ReflectJKnl(double *tnext);
__global__ void ReflectKKnl(double *tnext);


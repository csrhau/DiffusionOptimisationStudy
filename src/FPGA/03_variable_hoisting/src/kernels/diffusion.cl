// vim: set filetype=c:

#include "common.h"

__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1)))
krnl_temperature(__global float *restrict field,
                 __global float *restrict registers) {
  float cumsum = 0;
  for (unsigned k = 1; k < KMAX-1; ++k) {
    for (unsigned j = 1; j < JMAX-1; ++j) {
      for (unsigned i = 1; i < IMAX-1; ++i) {
        cumsum += field[k * JMAX * IMAX + j * IMAX + i];
      }
    }
  }
  registers[TEMPERATURE] = cumsum;
}

__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1)))
krnl_diffuse(__global float *restrict field_a,
             __global float *restrict field_b,
             float cx,
             float cy,
             float cz) {
  const size_t istride = 1;
  const size_t jstride = IMAX;
  const size_t kstride = IMAX*JMAX;
  float cumsum = 0;
  for (unsigned k = 1; k < KMAX-1; ++k) {
    for (unsigned j = 1; j < JMAX-1; ++j) {
      __attribute__((xcl_pipeline_loop))
      for (unsigned i = 1; i < IMAX-1; ++i) {
        const size_t center = k * JMAX * IMAX + j * IMAX + i;
        float field_center = field_a[center];
        float result = field_center
                     + cx * (field_a[center-istride] - 2*field_center + field_a[center+istride])
                     + cy * (field_a[center-jstride] - 2*field_center + field_a[center+jstride])
                     + cz * (field_a[center-kstride] - 2*field_center + field_a[center+kstride]);
        field_b[center] = result;
        // Reflective Boundary Condition
        if (i==1) {
          field_b[center-istride] = result;
        } else if (i == IMAX - 2) {
          field_b[center+istride] = result;
        } 
        if (j == 1) {
          field_b[center-jstride] = result;
        } else if (j == JMAX - 2) {
          field_b[center+jstride] = result;
        } 
        if (k == 1) {
          field_b[center-kstride] = result;
        } else if (k == KMAX - 2) {
          field_b[center+kstride] = result;
        } 
      }
    }
  }
}

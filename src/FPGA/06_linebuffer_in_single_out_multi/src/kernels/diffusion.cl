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
  float cumsum = 0;
  local float linebuf_in[IMAX];
  local float linebuf_out[IMAX];
  const size_t istride = 1;
  const size_t jstride = IMAX;
  const size_t kstride = IMAX*JMAX;
  for (unsigned k = 1; k < KMAX-1; ++k) {
    for (unsigned j = 1; j < JMAX-1; ++j) {
      const size_t line_start = k * JMAX * IMAX + j * IMAX;
      // TODO, COULD REDUCE GLOBAL OVERHEAD BY ANOTHER FACTOR OF 4
      async_work_group_copy(linebuf_in, field_a + line_start, IMAX, 0);
      barrier(CLK_LOCAL_MEM_FENCE);
      __attribute__((xcl_pipeline_loop))
      for (unsigned i = 1; i < IMAX-1; ++i) {
        const size_t center = line_start + i;
        float result = linebuf_in[i]
                     + cx * (linebuf_in[i-istride] - 2*linebuf_in[i] + linebuf_in[i+istride])
                     + cy * (field_a[center-jstride] - 2*linebuf_in[i] + field_a[center+jstride])
                     + cz * (field_a[center-kstride] - 2*linebuf_in[i] + field_a[center+kstride]);
        linebuf_out[i] = result;
        // Reflective Boundary Condition
        if (i==1) {
          linebuf_out[i-istride] = result;
        } else if (i == IMAX - 2) {
          linebuf_out[i+istride] = result;
        } 
      }
      async_work_group_copy(field_b + line_start, linebuf_out, IMAX, 0);
      // TODO pretty sure this could be done far better
      if (j == 1) {
         async_work_group_copy(field_b + line_start - jstride, linebuf_out, IMAX, 0);
      } else if (j == JMAX - 2) {
         async_work_group_copy(field_b + line_start + jstride, linebuf_out, IMAX, 0);
      } 
      if (k == 1) {
         async_work_group_copy(field_b + line_start - kstride, linebuf_out, IMAX, 0);
      } else if (k == KMAX - 2) {
         async_work_group_copy(field_b + line_start + kstride, linebuf_out, IMAX, 0);
      } 
    }
  }
}

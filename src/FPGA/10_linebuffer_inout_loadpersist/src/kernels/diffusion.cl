// vim: set filetype=c:
#include "common.h"

// Manages the circular buffer thing
#define LINEBUF_JPREV tempbuf[(j-1)%3]
#define LINEBUF_IN tempbuf[j%3]
#define LINEBUF_JNEXT tempbuf[(j+1)%3]

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
  local float planebuf[3][IMAX];
  local float tempbuf[3][IMAX];
  local float linebuf_kprev[IMAX];
  local float linebuf_knext[IMAX];
  local float linebuf_out[IMAX];
  const size_t istride = 1;
  const size_t kstride = IMAX*JMAX;
  for (unsigned k = 1; k < KMAX-1; ++k) {
    async_work_group_copy(tempbuf[0], field_a + k * JMAX * IMAX, IMAX, 0);
    async_work_group_copy(tempbuf[1], field_a + k * JMAX * IMAX + IMAX, IMAX, 0);
    for (unsigned j = 1; j < JMAX-1; ++j) {
      const size_t line_start = k * JMAX * IMAX + j * IMAX;
      async_work_group_copy(LINEBUF_JNEXT, field_a + line_start + IMAX, IMAX, 0);
      async_work_group_copy(linebuf_kprev, field_a + line_start - kstride, IMAX, 0);
      async_work_group_copy(linebuf_knext, field_a + line_start + kstride, IMAX, 0);
      barrier(CLK_LOCAL_MEM_FENCE);
      float left;
      float center = LINEBUF_IN[0];
      float right = LINEBUF_IN[1];
      for (unsigned i = 1; i < IMAX-1; ++i) {
        left = center;
        center = right;
        right = LINEBUF_IN[i+istride];
        float result = center 
                     + cx * (left - 2*center + right)
                     + cy * (LINEBUF_JPREV[i] - 2*center + LINEBUF_JNEXT[i])
                     + cz * (linebuf_kprev[i] - 2*center + linebuf_knext[i]);
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
         async_work_group_copy(field_b + line_start - IMAX, linebuf_out, IMAX, 0);
      } else if (j == JMAX - 2) {
         async_work_group_copy(field_b + line_start + IMAX, linebuf_out, IMAX, 0);
      } 
      if (k == 1) {
         async_work_group_copy(field_b + line_start - kstride, linebuf_out, IMAX, 0);
      } else if (k == KMAX - 2) {
         async_work_group_copy(field_b + line_start + kstride, linebuf_out, IMAX, 0);
      } 
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}

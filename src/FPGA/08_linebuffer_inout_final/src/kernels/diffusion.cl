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
  local float linebuf_jspan[3 * IMAX];
  local float linebuf_kprev[IMAX];
  local float linebuf_knext[IMAX];
  local float linebuf_out[IMAX];
  float iwindow[3];
  const size_t istride = 1;
  const size_t jstride = IMAX;
  const size_t kstride = IMAX*JMAX;
  for (unsigned k = 1; k < KMAX-1; ++k) {
    local float *linebuf_temp;
    local float *linebuf_jprev = linebuf_jspan;
    local float *linebuf_center = linebuf_jspan + IMAX;
    local float *linebuf_jnext = linebuf_jspan + IMAX + IMAX;
    async_work_group_copy(linebuf_jspan, field_a + k * kstride, IMAX * 2, 0);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned j = 1; j < JMAX-1; ++j) {
      const size_t line_start = k * JMAX * IMAX + j * IMAX;
      async_work_group_copy(linebuf_jnext, field_a + line_start + jstride, IMAX, 0);
      async_work_group_copy(linebuf_kprev, field_a + line_start - kstride, IMAX, 0);
      async_work_group_copy(linebuf_knext, field_a + line_start + kstride, IMAX, 0);
      barrier(CLK_LOCAL_MEM_FENCE);
      iwindow[0] = linebuf_center[0];
      iwindow[1] = linebuf_center[1];
      for (unsigned i = 1; i < IMAX-1; ++i) {
        iwindow[(i + 1) % 3] = linebuf_center[i + 1];
        float result = linebuf_center[i] 
                     + cx * (linebuf_center[i-istride] - 2*linebuf_center[i] + linebuf_center[i+istride])
                     + cy * (linebuf_jprev[i] - 2*linebuf_center[i] + linebuf_jnext[i])
                     + cz * (linebuf_kprev[i] - 2*linebuf_center[i] + linebuf_knext[i]);
        linebuf_out[i] = result;
        // Reflective Boundary Condition
        if (i==1) {
          linebuf_out[i-istride] = result;
        } else if (i == IMAX - 2) {
          linebuf_out[i+istride] = result;
        } 
      }
      // Move our pointers around the buffer
      linebuf_temp = linebuf_jprev;
      linebuf_jprev = linebuf_center;
      linebuf_center = linebuf_jnext;
      linebuf_jnext = linebuf_temp;
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
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}

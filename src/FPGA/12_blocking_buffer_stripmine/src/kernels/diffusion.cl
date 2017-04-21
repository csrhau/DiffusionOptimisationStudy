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
  const int istrips = (IMAX - 2 + ISTRIPWIDTH - 1) / ISTRIPWIDTH;
  const int jstrips = (JMAX - 2 + JSTRIPWIDTH - 1) / JSTRIPWIDTH;
  const int kstrips = (KMAX - 2 + KSTRIPWIDTH - 1) / KSTRIPWIDTH;

  local float inbuff[KMAX+2][JMAX+2][IMAX+2];
  for (int ks = 0; ks < kstrips; ++ks) {
    // TODO, account for non-perfect roundage
    const size_t k_start =  ks * KSTRIPWIDTH + 1;
    const size_t k_end = MIN((ks + 1) * KSTRIPWIDTH + 1, KMAX-1);
    for (int js = 0; js < jstrips; ++js) {
      const size_t j_start =  js * JSTRIPWIDTH + 1;
      const size_t j_end = MIN((js + 1) * JSTRIPWIDTH + 1, JMAX-1);
      for (int is = 0; is < istrips; ++is) {
        const size_t i_start =  is * ISTRIPWIDTH + 1;
        const size_t i_end = MIN((is + 1) * ISTRIPWIDTH + 1, IMAX-1);
        // Read in 
        for (unsigned k = k_start-1; k <= k_end; ++k) {
          for (unsigned j = j_start-1; j <= j_end; ++j) {
            for (unsigned i = i_start-1; i <= i_end; ++i) {
              inbuff[k][j][i] = field_a[k * JMAX * IMAX + j * IMAX + i];
            }
          }
        }
        // Process and emit
        for (unsigned k = k_start; k < k_end; ++k) {
          for (unsigned j = j_start; j < j_end; ++j) {
            for (unsigned i = i_start; i < i_end; ++i) {
              float result = inbuff[k][j][i] 
                           + cx * (inbuff[k][j][i-1] - 2*inbuff[k][j][i] + inbuff[k][j][i+1])
                           + cy * (inbuff[k][j-1][i] - 2*inbuff[k][j][i] + inbuff[k][j+1][i])
                           + cz * (inbuff[k-1][j][i] - 2*inbuff[k][j][i] + inbuff[k+1][j][i]);
              const int center = k * JMAX * IMAX + j * IMAX + i;
              field_b[center] = result;
              // Reflective Boundary Condition
              if (i==1) {
                field_b[center-1] = result;
              } else if (i == IMAX - 2) {
                field_b[center+1] = result;
              } 
              if (j == 1) {
                field_b[center-IMAX] = result;
              } else if (j == JMAX - 2) {
                field_b[center+IMAX] = result;
              } 
              if (k == 1) {
                field_b[center-IMAX*JMAX] = result;
              } else if (k == KMAX - 2) {
                field_b[center+IMAX*JMAX] = result;
              } 
            }
          }
        }
      }
    }
  }
}

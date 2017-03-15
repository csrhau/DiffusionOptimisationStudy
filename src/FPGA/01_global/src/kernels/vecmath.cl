// vim: set filetype=c:

__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1)))
krnl_vadd(__global float* x,
          __global float* y,
          __global float* z,
          unsigned n) {
  for (unsigned i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

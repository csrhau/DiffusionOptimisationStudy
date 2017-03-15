// vim: set filetype=c:

__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1)))
krnl_ivadd(__global int* x,
          __global int* y,
          __global int* z,
          const int length) {
  for (int i = 0; i < length; ++i) {
    z[i] = x[i] + y[i];
  }
}

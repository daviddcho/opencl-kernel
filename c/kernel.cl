// OpenCL Kernel
__kernel void mmul(__global float *a, __global float *b, __global float *c, const int N) {
  int k;
  int i = get_global_id(0);
  int j = get_global_id(1);

  float tmp = 0.0;
  for (k = 0; k < N; k++) {
    tmp += a[j*N+k] * b[k*N+i]; 
  }
  c[j*N+i] = tmp;
}


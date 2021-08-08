
__kernel void mmul(const int N, __global float *a, __global float *b, __global float *c, __local float* Bwrk) {
  int k, j; 
  int i = get_global_id(0);
  int iloc = get_local_id(0);
  int nloc = get_local_size(0);
  float Awrk[1024];
  float tmp;

  if (i < N) {
    for (k = 0; k < N; k++) {
      Awrk[k] = a[i*N+k];
    }
    for (j = 0; j < N; j++) {
      barrier(CLK_LOCAL_MEM_FENCE);
      for (k = iloc; k < N; k += nloc)
        Bwrk[k] = b[k*N+j];
      barrier(CLK_LOCAL_MEM_FENCE);
      tmp = 0.0f;
      for (k = 0; k < N; k++) 
        tmp += Awrk[k] * Bwrk[k];
      c[i*N+j] = tmp;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}


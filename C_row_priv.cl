
__kernel void mmul(const int N, __global float *a, __global float *b, __global float *c) {
  int i = get_global_id(0);
  int k, j;
  float tmp;
  float Awrk[N];
  
  if ((i < N)) {
    for (k = 0; k < N; k++) {
      Awrk[k] = a[i*N+k];
    }
    for (int j = 0; j < N; j++) {
      tmp = 0.0f;
      for (k = 0; k < N; k++) {
        tmp += Awrk[k] * b[k * N + j];
      }
      c[i * N + j] = tmp;
    }
  }
}

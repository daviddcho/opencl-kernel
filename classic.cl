
__kernel void mmul(const int N, __global float *a, __global float *b, __global float *c) {
  
  int i = get_global_id(0);
  int j = get_global_id(1);
  int k;
  float tmp;
  
  if ((i < N) && (j < N)) {
    tmp = 0.0f;
    for (k = 0; k < N; k++) {
      tmp += a[i * N + k] * b[k * N + j];
    }
    c[i * N + j] = tmp;
  }
}

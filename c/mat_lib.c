// Set matrix to zero
void zero_mat(int N, float *C) {
  int i, j;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) 
      C[i * N + j] = 0.0f; 
  }
}



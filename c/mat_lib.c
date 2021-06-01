#include <stdio.h>

// Sequential matrix multiplication
void sequential_mat_mul(float *A, float *B, float *C, int N) {
  int i,j,k;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      float tmp = 0.0f;
      for (k = 0; k < N; k++) {
        tmp += A[i * N + k] * B[k * N + j]; 
      }
      C[i * N + j] = tmp;
    }
  }
}

// Prints matrix
void print_mat(float *A, int N) {
  for (int i = 0; i < N*N; i++) {
    printf("%f ", A[i]);
    if (((i+1)%N) == 0 && i != 0) {
      //printf("%d", i);
      printf("\n");
    }
  }
  printf("\n");
}

// Set matrix to zero
void zero_mat(float *C, int N) {
  int i, j;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) 
      C[i * N + j] = 0.0f; 
  }
}


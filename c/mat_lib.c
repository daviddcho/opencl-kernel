#include <stdio.h>
#include <stdlib.h>

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

char* load_kernel(char *filename) {
  FILE *fp;
  long lSize;
  char *buffer;

  fp = fopen(filename, "rb");
  if (!fp) {
    perror("failed opening file");
    exit(1);
  }

  fseek(fp, 0L, SEEK_END);
  lSize = ftell(fp);
  rewind(fp);

  printf("%ld\n", lSize);

  buffer = calloc(sizeof(char), lSize+1);
  if (!buffer) {
    fclose(fp);
    fputs("memory alloc failed", stderr);
    exit(1);
  }

  if (sizeof(char) != fread(buffer, lSize, sizeof(char), fp)) {
    fclose(fp);
    free(buffer);
    fputs("entire read failed", stderr);
    exit(1);
  }

  fclose(fp);
  return buffer;
}




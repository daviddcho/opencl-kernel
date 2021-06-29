#ifndef MAT_LIB
#define MAT_LIB

#include <stdio.h>
#include <stdlib.h>

void sequential_mat_mul(float *A, float *B, float *C, int N);
void zero_mat(float *C, int N);
char* load_kernel(char* filename);

#endif 

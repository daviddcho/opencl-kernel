/*
 * Matrix multiplication (c = a * b)
*/

#include<stdio.h>
#include<stdlib.h>
#include<sys/types.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"
#include "mat_lib.h"

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

extern double wtime();
extern int output_device_info(cl_device_id);
void print_mat(float* A, int N);


#define TOL   (0.0001)
#define N     (64)

/*
const char *kernel_source = "\n" \
"__kernel void mmul(__global float *a, __global float *b, __global float *c, const int N) {\n" \
"  int i = get_global_id(0);                         \n" \
"  int j = get_global_id(1);                         \n" \
"  int k;                                            \n" \
"  float tmp;                                        \n" \
"                                                    \n" \
"  if ((i < N) && (j < N)) {                         \n" \
"    tmp = 0.0f;                                     \n" \
"    for (k = 0; k < N; k++) {                       \n" \
"      tmp += a[j * N + k] * b[k * N + i];           \n" \
"    }                                               \n" \
"    c[j * N + i] = tmp;                             \n" \
"  }                                                 \n" \
"}                                                    \n" \
"\n";
*/



int main(int argc, char** argv) { 
  int err;
  int size = N*N;
  const char *kernel_source = load_kernel("kernel.cl");

  float* h_a = (float *) calloc(size, sizeof(float));
  float* h_b = (float *) calloc(size, sizeof(float));
  float* h_c = (float *) calloc(size, sizeof(float));

  //size_t global; 
  cl_device_id device_id;
  cl_context context;
  cl_command_queue commands;
  cl_program program;
  cl_kernel ko_mmul;

  cl_mem d_a;
  cl_mem d_b;
  cl_mem d_c;

  srand(42);

  // Fill in matrices
  int i;
  int count = size;
  for (i = 0; i < count; i++) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
  }
  

  sequential_mat_mul(h_a, h_b, h_c, N);
  printf("Sequential C\n");
  print_mat(h_c, N);
  

  // Set up platform and GPU device
  cl_uint numPlatforms;
  err = clGetPlatformIDs(0, NULL, &numPlatforms);
  checkError(err, "Finding platforms");
  if (numPlatforms == 0) {
    printf("Found 0 platforms\n");
    return EXIT_FAILURE;
  }

  // Get all platforms 
  cl_platform_id Platform[numPlatforms];
  err = clGetPlatformIDs(numPlatforms, Platform, NULL);
  checkError(err, "Getting platforms");

  // Secure a GPU
  for (i = 0; i < numPlatforms; i++) {
    err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
    if (err == CL_SUCCESS) {
      break;
    }
  }
  
  if (device_id == NULL)
    checkError(err, "Finding a device");
  
  err = output_device_info(device_id);
  checkError(err, "Finding device output");

  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  checkError(err, "Creating context");

  commands = clCreateCommandQueue(context, device_id, 0, &err);
  checkError(err, "Creating command queue");


  //load_kernel("kernel.cl", &kernel_source);

  program = clCreateProgramWithSource(context, 1, (const char**) &kernel_source, NULL, &err);
  checkError(err, "Creating program");

  // Build the program
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];

    printf("Error: Failed to build program executablen\n%s\n", err_code(err));
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    return EXIT_FAILURE;
  }

  ko_mmul = clCreateKernel(program, "mmul", &err);
  checkError(err, "Creating kernel");

  // Create the input and output arrays in device memory
  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) *count, NULL, &err);
  checkError(err, "Creating buffer d_a");
  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) *count, NULL, &err);
  checkError(err, "Creating buffer d_b");
  d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) *count, NULL, &err);
  checkError(err, "Creating buffer d_c");

  // Write vectors into compute device memory
  err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float) *count, h_a, 0, NULL, NULL);
  checkError(err, "Copying h_a to device at d_a");
  err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(float) *count, h_b, 0, NULL, NULL);
  checkError(err, "Copying h_b to device at d_b");
  

  // Set the arguments to our compute kernel
  err = clSetKernelArg(ko_mmul, 0, sizeof(cl_mem), &d_a);
  err |= clSetKernelArg(ko_mmul, 1, sizeof(cl_mem), &d_b);
  err |= clSetKernelArg(ko_mmul, 2, sizeof(cl_mem), &d_c);
  err |= clSetKernelArg(ko_mmul, 3, sizeof(unsigned int), &count);
  checkError(err, "Setting kernel arguments");

  double rtime = wtime();
  
  // Execute the kernel
  const size_t global_work_size[2] = {N, N};
  //const size_t local_work_size[2] = {16, 16};
  //global = count;
  err = clEnqueueNDRangeKernel(commands, ko_mmul, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
  checkError(err, "Enqueueing kernel"); 

  err = clFinish(commands);
  checkError(err, "Waiting for kernel to finish");

  rtime = wtime() - rtime;
  printf("\nThe kernel ran in %lf seconds\n", rtime);

  // Read back the results from compute device
  err = clEnqueueReadBuffer(commands, d_c, CL_TRUE, 0, sizeof(float) *count, h_c, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: failed to read output array!\n%s\n", err_code(err));
    exit(1);
  }
  
  printf("A:\n");
  print_mat(h_a, N);
  printf("B:\n");
  print_mat(h_b, N);
  printf("C:\n");
  print_mat(h_c, N);
  
  //printf("C = A*B: %d out of %d results were correct", test_results(h_a, h_b, h_c, count), count);

  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseProgram(program);
  clReleaseKernel(ko_mmul);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  free(h_a);
  free(h_b);
  free(h_c);
  
  return 0;
}




/*
 * Addition of two vectors (c = a + b) 
 * CHAINING: c = a + b, d = c + e, f = d + g
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

// Pick up device type from compiler commd line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

extern double wtime();
extern int output_device_info(cl_device_id);

#define TOL     (0.001) // Tolerance used inf loating point comparisons
#define LENGTH  (1024)  // Length of vectors a, b, and c

int test_results(float* h_a, float* h_b, float* h_c, int count);

const char *KernelSource = "\n" \
"__kernel void vadd(                      \n" \
" __global float* a,                      \n" \
" __global float* b,                      \n" \
" __global float* c,                      \n" \
" const unsigned int count) {             \n" \
"   int i = get_global_id(0);             \n" \
"   if (i < count)                        \n" \
"     c[i] = a[i] + b[i];                 \n" \
"}                                        \n" \
"\n";


int main(int argc, char** argv) {
  int err;

  float* h_a = (float*) calloc(LENGTH, sizeof(float));
  float* h_b = (float*) calloc(LENGTH, sizeof(float));
  float* h_c = (float*) calloc(LENGTH, sizeof(float));
  float* h_d = (float*) calloc(LENGTH, sizeof(float));
  float* h_e = (float*) calloc(LENGTH, sizeof(float));
  float* h_f = (float*) calloc(LENGTH, sizeof(float));
  float* h_g = (float*) calloc(LENGTH, sizeof(float));
  
  // Number of correct results
  //unsigned int correct;
  
  // Global domain size
  size_t global;
  
  cl_device_id device_id;
  cl_context context;
  cl_command_queue commands;
  cl_program program;
  cl_kernel ko_vadd;            // Compute kernel
  
  // Device memory used for vectors
  cl_mem d_a; 
  cl_mem d_b; 
  cl_mem d_c;
  cl_mem d_d;
  cl_mem d_e;
  cl_mem d_f;
  cl_mem d_g;

  // Fill vectors a and b with random float values
  int i = 0;
  int count = LENGTH;
  for (i = 0; i < count; i++) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
    h_e[i] = rand() / (float)RAND_MAX;
    h_g[i] = rand() / (float)RAND_MAX;
  }

  // Set up platform and GPU device
  
  cl_uint numPlatforms;
  err = clGetPlatformIDs(0, NULL, &numPlatforms);
  checkError(err, "Finding platforms");
  if (numPlatforms == 0) {
    printf("Found 0 platforms!\n");
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

  // Create a compute context
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  checkError(err, "Creating context");

  // Create a command queue
  commands = clCreateCommandQueue(context, device_id, 0, &err);
  checkError(err, "Creating command queue");

  // Create the compute program from the source buffer
  program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
  checkError(err, "Creating program");

  // Build the program 
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len; 
    char buffer[2048];

    printf("Error: Failed to build program executable!\n%s\n", err_code(err));
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    return EXIT_FAILURE;
  }

  // Create the compute kernel from the program
  ko_vadd = clCreateKernel(program, "vadd", &err);
  checkError(err, "Creating kernel");

  // Create the input (a, b, e, g) and output (c, d, f) arrays in device memory
  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, &err);
  checkError(err, "Creating buffer d_a");
  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, &err);
  checkError(err, "Creating buffer d_b");
  d_e = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, &err);
  checkError(err, "Creating buffer d_e");
  d_g = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, &err);
  checkError(err, "Creating buffer d_g");

  d_c = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * count, NULL, &err);
  checkError(err, "Creating buffer d_c");
  d_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * count, NULL, &err);
  checkError(err, "Creating buffer d_c");
  d_f = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, &err);
  checkError(err, "Creating buffer d_c");

  // Write vectors into compute device memory 
  err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float) * count, h_a, 0, NULL, NULL);
  checkError(err, "Copying h_a to device at d_a");
  err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(float) * count, h_b, 0, NULL, NULL);
  checkError(err, "Copying h_b to device at d_b");
  err = clEnqueueWriteBuffer(commands, d_e, CL_TRUE, 0, sizeof(float) * count, h_e, 0, NULL, NULL);
  checkError(err, "Copying h_e to device at d_e");
  err = clEnqueueWriteBuffer(commands, d_g, CL_TRUE, 0, sizeof(float) * count, h_g, 0, NULL, NULL);
  checkError(err, "Copying h_g to device at d_g");

  // Set the arguments to our compute kernel
  err = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_a);
  err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_b);
  err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
  err |= clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &count);
  checkError(err, "Setting kernel arguments");

  double rtime = wtime();

  // Execute the kernel over the entire range of our 1d input data 
  // letting the OpenCL runtime choose the work-group size
  global = count;
  err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
  checkError(err, "Enqueueing kernel");
  
  // d = c + e
  err = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_c);
  err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_e);
  err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_d);
  err |= clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &count);
  checkError(err, "Setting kernel arguments");

  err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
  checkError(err, "Enqueueing kernel");
  
  // f = d + g
  err = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_d);
  err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_g);
  err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_f);
  err |= clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &count);
  checkError(err, "Setting kernel arguments");

  err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
  checkError(err, "Enqueueing kernel");

  // Wait for the commands to complete before stopping the timer
  err = clFinish(commands);
  checkError(err, "Waiting for kernel to finish");

  rtime = wtime() - rtime;
  printf("\nThe kernel ran in %lf seconds\n", rtime);
  
  // Read back the results from the compute device 
  err = clEnqueueReadBuffer(commands, d_c, CL_TRUE, 0, sizeof(float) * count, h_c, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to read output array!\n%s\n", err_code(err));
    exit(1);
  }
  err = clEnqueueReadBuffer(commands, d_d, CL_TRUE, 0, sizeof(float) * count, h_d, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to read output array!\n%s\n", err_code(err));
    exit(1);
  }
  err = clEnqueueReadBuffer(commands, d_f, CL_TRUE, 0, sizeof(float) * count, h_f, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to read output array!\n%s\n", err_code(err));
    exit(1);
  }

  /*
  // Test the results 
  correct = 0;
  float tmp;

  for (i = 0; i < count; i++) {
    tmp = h_a[i] + h_b[i];
    tmp -= h_c[i];
    // Correct if square deviation is less than tolerance squared
    if (tmp*tmp < TOL*TOL) {
      correct++;
    } else {
      printf(" tmp %f h_a %f h_b %f h_c %f \n", tmp, h_a[i], h_b[i], h_c[i]);
    }
  }
  */

  // Summarize results
  printf("C = A+B: %d out of %d results were correct.\n", test_results(h_a, h_b, h_c, count), count);
  printf("D = C+E: %d out of %d results were correct.\n", test_results(h_c, h_e, h_d, count), count);
  printf("F = D+G: %d out of %d results were correct.\n", test_results(h_d, h_g, h_f, count), count);

  // Clean up 
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseMemObject(d_d);
  clReleaseMemObject(d_e);
  clReleaseMemObject(d_f);
  clReleaseMemObject(d_g);
  clReleaseProgram(program);
  clReleaseKernel(ko_vadd);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);
  
  free(h_a);
  free(h_b);
  free(h_c);
  free(h_d);
  free(h_e);
  free(h_f);
  free(h_g);

  return 0;
}

// Test the results
int test_results(float* h_a, float* h_b, float* h_c, int count) {
  unsigned int correct = 0;
  float tmp;
  for (int i = 0; i < count; i++) {
    tmp = h_a[i] + h_b[i];
    tmp -= h_c[i];
    // Correct if square deviation is less than tolerance squared
    if (tmp*tmp < TOL*TOL) {
      correct++;
    } else {
      printf(" tmp %f h_a %f h_b %f h_c %f \n", tmp, h_a[i], h_b[i], h_c[i]);
    }
  }
  return correct;
}

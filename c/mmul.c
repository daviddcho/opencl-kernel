/*
 * Matrix multiplication (c = a * b)
*/

#include<stdio.h>
#include<stdlib.h>
#include<sys.types.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFUALT
#endif

extern double wtime();
extern int output_device_info(cl_device_id);

#define TOL   (0.0001)
#define N     (1024)

const char *kernal_source = ""

int main(int argc, char** argv) {


}


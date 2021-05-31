import pyopencl as cl
import numpy

import deviceinfo 
from helper import *
from time import time

kernelsource = """
__kernel void mmul(__global float *a, __global float *b, __global float *c, const int N) {
  
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
"""

N = LENGTH 
size = N*N

# Create a compute context
context = cl.create_some_context() 

# Print out device info
deviceinfo.output_device_info(context.devices[0])

# Create a command queue
queue = cl.CommandQueue(context)

# Create the compute program from the source buffer and build it
program = cl.Program(context, kernelsource).build() 

h_a = numpy.random.rand(size).astype(numpy.float32) 
h_b = numpy.random.rand(size).astype(numpy.float32) 
h_c = numpy.empty(size).astype(numpy.float32)

d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a) 
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_c.nbytes)

# Start the timer
start_time = time() 

mmul = program.mmul
# This is basically for the kernel args: None for the globals and specified for the non global I think
mmul.set_scalar_arg_dtypes([None, None, None, numpy.uint32])

# Execute 
globalrange = (N, N)
localrange = None
mmul(queue, globalrange, localrange, d_a, d_b, d_c, N)
queue.finish() 

run_time = time() - start_time
#print("The kernel ran in %f seconds" % run_time)

# Read back the results from the compute device
cl.enqueue_copy(queue, h_c, d_c)

# Do the calculation on the host for comparison
# TODO: Testing could be better, maybe a test set or something?
h_test = numpy.empty(size).astype(numpy.float32)
#sequential(N, h_a, h_b, h_test)

#print(h_test) 
print(h_c)

results(N, h_c, h_test, run_time)


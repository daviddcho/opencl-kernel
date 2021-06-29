import pyopencl as cl
import numpy as np

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

# row using private memory now?
# i kinda forget how this weird vector mult works
useprivatemem = """
__kernel void mmul(__global float *a, __global float *b, __global float *c, const int N) {
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
program = cl.Program(context, useprivatemem).build() 

h_a = np.random.rand(size).astype(np.float32) 
h_b = np.random.rand(size).astype(np.float32) 
h_c = np.empty(size).astype(np.float32)

d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a) 
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_c.nbytes)

# Start the timer
start_time = time() 

# This is basically for the kernel args: 
# None for the globals and specified for the non global I think
mmul = program.mmul
mmul.set_scalar_arg_dtypes([None, None, None, np.uint32])

# Execute 
#globalrange = (N, N)
globalrange = (N,)
localrange = None


#localmem = cl.LocalMemory(np.dtype(np.float32).itemsize * globalrange)
mmul(queue, globalrange, localrange, d_a, d_b, d_c, N)

queue.finish() 

run_time = time() - start_time
#print("The kernel ran in %f seconds" % run_time)

# Read back the results from the compute device
cl.enqueue_copy(queue, h_c, d_c)

# Do the calculation on the host for comparison
# TODO: Testing could be better, maybe a test set or something?
C = np.empty(size).astype(np.float32)
#sequential(N, h_a, h_b, C)

print(h_c)

results(N, h_c, C, run_time)
#assert np.allclose(h_c, C)


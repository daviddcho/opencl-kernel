#!/usr/bin/env python3
import pyopencl as cl
import numpy as np

import deviceinfo 
from helper import *
from time import time

with open("classic.cl", "r") as file:
  kernelsource = file.read()

N = LENGTH
size = N*N

# Create a compute context
context = cl.create_some_context() 
deviceinfo.output_device_info(context.devices[0])

queue = cl.CommandQueue(context)
program = cl.Program(context, kernelsource).build() 

h_a = np.random.rand(size).astype(np.float32) 
h_b = np.random.rand(size).astype(np.float32) 
h_c = np.empty(size).astype(np.float32)

d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a) 
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_c.nbytes)

start_time = time() 

mmul = program.mmul

"""
# Using local memory
mmul.set_scalar_arg_dtypes([np.uint32, None, None, None, None])
globalrange = (N, N)
localrange = None
localmem = cl.LocalMemory(np.dtype(np.float32).itemsize * N)
mmul(queue, globalrange, localrange, N, d_a, d_b, d_c, localmem)
"""

mmul.set_scalar_arg_dtypes([np.uint32, None, None, None])
globalrange = (N, N)
localrange = None
mmul(queue, globalrange, localrange, N, d_a, d_b, d_c)

queue.finish() 

run_time = time() - start_time
#print("The kernel ran in %f seconds" % run_time)

# Read back the results from the compute device
cl.enqueue_copy(queue, h_c, d_c)

print(h_c)
C = np.empty(size).astype(np.float32)
sequential(N, h_a, h_b, C)

flops(N, run_time)
assert np.allclose(h_c, C)

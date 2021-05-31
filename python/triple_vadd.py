import pyopencl as cl 
import numpy 

import deviceinfo

from time import time

TOL = 0.001
LENGTH = 1024

kernelsource = """
__kernel void vadd(__global float *a, __global float *b, __global float *c, __global float *d, const unsigned int count) {
  int i = get_global_id(0);
  if (i < count) {
    d[i] = a[i] + b[i] + c[i];
  }
}
"""

# Create a compute context
context = cl.create_some_context() 

# Print out device info
deviceinfo.output_device_info(context.devices[0])

# Create a command queue
queue = cl.CommandQueue(context)

# Create the compute program from the source buffer and build it 
program = cl.Program(context, kernelsource).build()

h_a = numpy.random.rand(LENGTH).astype(numpy.float32)
h_b = numpy.random.rand(LENGTH).astype(numpy.float32) 
h_c = numpy.random.rand(LENGTH).astype(numpy.float32)
h_d = numpy.empty(LENGTH).astype(numpy.float32) 

d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)
d_c = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_c)
d_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_d.nbytes)

# Start the timer
rtime = time()

# Initiate the kernel
vadd = program.vadd
vadd.set_scalar_arg_dtypes([None, None, None, None, numpy.uint32])

# Execute
vadd(queue, h_a.shape, None, d_a, d_b, d_c, d_d, LENGTH)
queue.finish()

rtime = time() - rtime 
print("The kernel ran in %f seconds" % rtime)

# Read back the results from the compute device
cl.enqueue_copy(queue, h_d, d_d)

# Test results
correct = 0
for i in range(LENGTH): 
  expected = h_a[i] + h_b[i] + h_c[i]
  actual = h_d[i]
  # Comnpute the relative error
  rel_err = numpy.absolute((actual - expected) / expected)
  if rel_err < TOL:
    correct += 1
  else:
    print("i: %d is wrong. rel_err %f" % (i, relative_error))

print(h_d)
print("D = A+B+C: %d out of %d results wer correct" % (correct, LENGTH))

"""
# Some advanced way of testing I don't understand 
correct = 0 
for a, b, c, d in zip(h_a, h_b, h_c, h_d):
  tmp = a + b + c
  tmp -= d
  if tmp*tmp < TOL*TOL:
    correct += 1
  else:
    print("tmp %f h_a %f h_b %f h_c %f h_d %f" % (tmp, a, b, c, d))
"""


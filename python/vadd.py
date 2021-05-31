import pyopencl as cl
import numpy

import deviceinfo

from time import time

TOL = 0.001
LENGTH = 1024

kernelsource = """
__kernel void vadd(__global float *a, __global float *b, __global float *c, const unsigned int count) {
  int i = get_global_id(0);
  if (i < count) {
    c[i] = a[i] + b[i];
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

# Create vectors and fill with random float values 
h_a = numpy.random.rand(LENGTH).astype(numpy.float32) 
h_b = numpy.random.rand(LENGTH).astype(numpy.float32) 
h_c = numpy.random.rand(LENGTH).astype(numpy.float32)
h_d = numpy.random.rand(LENGTH).astype(numpy.float32)
h_e = numpy.random.rand(LENGTH).astype(numpy.float32) 
h_f = numpy.random.rand(LENGTH).astype(numpy.float32)
h_g = numpy.random.rand(LENGTH).astype(numpy.float32)

# Create the input (a, b, e, g) arrays in device memory and copy data from host
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a) 
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b) 
d_e = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_e) 
d_g = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_g)
# Create the output c, d, f array in device memory
d_c = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_c.nbytes)
d_d = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_d.nbytes) 
d_f = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_f.nbytes) 

# Start the timer
rtime = time()

# Execute the kernel over the enture range of our 1d input
# allowing OpenCL runtime to select the work group items for the device
vadd = program.vadd
vadd.set_scalar_arg_dtypes([None, None, None, numpy.uint32])

# Add vectors to the queue to do their thing
vadd(queue, h_a.shape, None, d_a, d_b, d_c, LENGTH) 
vadd(queue, h_a.shape, None, d_c, d_e, d_d, LENGTH)
vadd(queue, h_a.shape, None, d_d, d_g, d_f, LENGTH)

# Wait for queue to finish before reading
queue.finish()
rtime = time() - rtime
print("The kernel ran in %f seconds" % rtime)

# Read back the results from the compute device 
cl.enqueue_copy(queue, h_c, d_c)
cl.enqueue_copy(queue, h_d, d_d)
cl.enqueue_copy(queue, h_f, d_f)

#print(h_f)

# Test results
def test(h_a, h_b, h_c):
  correct = 0
  for a, b, c in zip(h_a, h_b, h_c):
    tmp = a + b
    tmp -= c
    # Correct if square deviation is less than tolerance squared 
    if tmp*tmp < TOL*TOL:
      correct += 1
    else:
     print("tmp %f h_a %f h_b %f h_c %f" % (tmp, a, b, c))
  return correct

# Summary results
print("C = A+B: %d out of %d results were correct" % (test(h_a, h_b, h_c), LENGTH))
print("D = C+E: %d out of %d results were correct" % (test(h_c, h_e, h_d), LENGTH))
print("F = D+G: %d out of %d results were correct" % (test(h_g, h_d, h_f), LENGTH))



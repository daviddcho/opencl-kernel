import numpy

TOL = 0.0001
LENGTH = 16
AVAL, BVAL = 3.0, 5.0

# Function to compute the matrix product
def sequential(N, a, b, c):
  # Iterate over the rows of A
  for i in range(N):
    # Iterate over the columbs of B 
    for j in range(N):
      tmp = 0.0
      for k in range(N):
        tmp += a[i * N + k] * b[k * N + j]
      c[i * N + j] = tmp 

# Function to compute errors of the product matrix
def error(N, c, c2):
  for i in range(N*N):
    expected = c2[i]
    actual = c[i]
    rel_err = numpy.absolute((actual - expected) / expected)
    if rel_err > TOL:
      return -1 
  return 
"""
  cval = float(N) * AVAL * BVAL 
  errsq = 0.0
  for i in range(N):
    for j in range(N):
      err = c[i*N+j] - cval
      errsq += err*err
  return errsq
"""

# Function to analyze and output results
def results(N, c, c2, run_time):
  mflops = 2.0 * N * N * N/(1000000.0*run_time)
  print("matrix size %d: %f seconds at %f MFLOPS" % (N, run_time, mflops))

  """
  if error(N, c, c2) == -1:
    print("Error in matrix product")
  else:
    print("Passed error check")
  """

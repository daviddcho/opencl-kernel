TOL = 0.0001
LENGTH = 16

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

# im not sure if this flops thing is right
def flops(N, run_time):
  mflops = 2.0 * N * N * N/(1000000.0*run_time)
  print("matrix size %d: %f seconds at %f MFLOPS" % (N, run_time, mflops))



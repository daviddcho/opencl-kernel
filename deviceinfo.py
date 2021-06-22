import pyopencl as cl
import sys

def output_device_info(device_id):
  sys.stdout.write("Device is ")
  sys.stdout.write("device_id.name")
  if device_id.type == cl.device_type.GPU:
    sys.stdout.write("GPU from ") 
  elif device_id.type == cl.device_type.CPU:
    sys.stdout.write("CPU from ")
  else:
    sys.stdout.write("non CPU of CPU processor from ")

  sys.stdout.write(device_id.vendor)
  sys.stdout.write(" with a max of ")
  sys.stdout.write(str(device_id.max_compute_units))
  sys.stdout.write(" compute units\n") 
  sys.stdout.flush()

import pyopencl as cl
import numpy as np

def get_intel_device():
    platforms = cl.get_platforms()
    for platform in platforms:
        if "Intel" in platform.name:
            print(f"Found Intel platform: {platform.name}")
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if devices:
                return devices[0]
            else:
                print("No GPU devices found on the Intel platform.")
                raise RuntimeError("No GPU found on Intel OpenCL platform.")
    raise RuntimeError("Intel OpenCL platform not found.")

device = get_intel_device()
print("Using device:", device.name)

context = cl.Context([device])
queue = cl.CommandQueue(context)

kernel_code = """
__kernel void duplicate_matrix(__global const float *input_matrix,
                               __global float *output_matrix,
                               const int width, const int height) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < height && col < width) {
        int index = row * width + col;
        output_matrix[index] = input_matrix[index];
    }
}
"""

program = cl.Program(context, kernel_code).build()

input_matrix = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]], dtype=np.float32)
height, width = input_matrix.shape

mf = cl.mem_flags
input_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_matrix)
output_buffer = cl.Buffer(context, mf.WRITE_ONLY, input_matrix.nbytes)

# Correct way to get the kernel - works for newer pyopencl versions
try:
    kernel = program.duplicate_matrix
except AttributeError:  # For older pyopencl versions
    kernel = cl.Kernel(program, "duplicate_matrix")  # Correct fallback

kernel.set_arg(0, input_buffer)
kernel.set_arg(1, output_buffer)
kernel.set_arg(2, np.int32(width))
kernel.set_arg(3, np.int32(height))

global_size = (height, width)
local_size = None
cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)

output_matrix = np.empty_like(input_matrix)
cl.enqueue_copy(queue, output_matrix, output_buffer).wait() # Add .wait() to ensure copy finishes before printing

queue.finish()

print("Input matrix:")
print(input_matrix)
print("\nDuplicated matrix:")
print(output_matrix)
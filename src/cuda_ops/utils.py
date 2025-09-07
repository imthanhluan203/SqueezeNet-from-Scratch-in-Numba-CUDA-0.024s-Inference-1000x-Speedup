"""
Utility CUDA operations
"""
import numpy as np
from numba import cuda

def run_concat_cuda(x_np_1, x_np_2, TPB=16):
    """Run concatenation on CUDA"""
    B, C, H, W = x_np_1.shape
    C_out = C*2
    @cuda.jit(cache=True)
    def Concat_Cuda(inp_1, inp_2, out):
        # merge batch và channel vào blockIdx.z
        z = cuda.blockIdx.z
        b = z // C_out
        c = z % C_out
    
        # tọa độ output
        x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        if y < H and x < W:
            if c < C:
                out[b, c, y, x] = inp_1[b, c, y, x]
            else:
                out[b, c, y, x] = inp_2[b, c-C, y, x]
    
    d_out = cuda.device_array((B, C_out, H, W), dtype=np.float32)
    bpg_x = (H + TPB - 1) // TPB
    bpg_y = (W + TPB - 1) // TPB
    bpg_z = B * C_out
     
    grid_dims = (bpg_x, bpg_y, bpg_z)
    block_dims = (TPB, TPB, 1)

    Concat_Cuda[grid_dims, block_dims](
        x_np_1, x_np_2, d_out
    )
    cuda.synchronize()
    return d_out

def run_sum_matrix(inp_1, inp_2, TPB=16):
    """Run matrix sum on CUDA"""
    B, C, H, W = inp_1.shape
    bpx = (H + TPB - 1)//TPB
    bpy = (W + TPB - 1)//TPB
    @cuda.jit(cache=True)
    def sum_matrix(matrix_1, matrix_2, out):
        x = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
        y = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y
        z = cuda.blockIdx.z
        b = z//C
        c = z%C
        if x < H and y < W:
            out[b, c, y, x] = matrix_1[b, c, y, x] + matrix_2[b, c, y, x]
    
    dout = cuda.device_array((B, C, H, W), dtype=np.float32)
    bpz = B*C
    griddim = (bpx, bpy, bpz)
    blockdim = (TPB, TPB, 1)
    sum_matrix[griddim, blockdim](inp_1, inp_2, dout)
    cuda.synchronize()
    return dout

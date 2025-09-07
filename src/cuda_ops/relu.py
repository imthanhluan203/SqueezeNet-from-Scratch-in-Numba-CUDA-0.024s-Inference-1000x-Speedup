"""
CUDA operations for ReLU
"""
import numpy as np
from numba import cuda

def run_Relu_Cuda(x_relu, TPB=16):
    """Run ReLU activation on CUDA"""
    B, C, H, W = x_relu.shape
    # Chuẩn bị output trên GPU
    d_out = cuda.device_array((B, C, H, W), dtype=np.float32)
    bpg_x = (H + TPB - 1) // TPB
    bpg_y = (W + TPB - 1) // TPB
    bpg_z = B * C
    
    @cuda.jit(cache=True)
    def Relu_Cuda(inp, out):
        z = cuda.blockIdx.z
        b = z // C
        c = z % C
        # tọa độ output
        x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        if y < H and x < W:
            out[b, c, y, x] = inp[b, c, y, x] if inp[b, c, y, x] > 0 else 0
    
    grid_dims = (bpg_x, bpg_y, bpg_z)
    block_dims = (TPB, TPB, 1)
    Relu_Cuda[grid_dims, block_dims](
        x_relu, d_out
    )
    cuda.synchronize()
    return d_out

def run_relu_backward(X, LO, TPB=16):
    """Run ReLU backward on CUDA"""
    B, C, H, W = LO.shape
    @cuda.jit(cache=True)
    def relu_backward(inp, lo, output):
        z = cuda.blockIdx.z
        b = z//C
        c = z%C
        x = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
        y = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y
        if x < H and y < W:
            if inp[b, c, y, x] < 0:
                output[b, c, y, x] = 0
            else:
                output[b, c, y, x] = lo[b, c, y, x]
    
    d_out = cuda.device_array((B, C, H, W), dtype=np.float32)
    bpg_x = (H + TPB - 1) // TPB
    bpg_y = (W + TPB - 1) // TPB
    bpg_z = B*C
    griddim = (bpg_x, bpg_y, bpg_z)
    blockdim = (TPB, TPB, 1)
    relu_backward[griddim, blockdim](X, LO, d_out)
    cuda.synchronize()
    return d_out
